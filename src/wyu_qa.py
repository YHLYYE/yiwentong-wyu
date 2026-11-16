"""邑问通：BM25 + Faiss 混合召回 + 三态判定 + 引用 + 无据拒答。

设计目标：
- 默认走完整链路（BM25 + bge-m3 + Faiss + DeepSeek）；
- 没有 GPU 时自动用 CPU；
- 没有模型/没有 API key 时降级为 BM25-only / 离线原文摘录，保证可复现、可演示。
"""
from __future__ import annotations

import os
import pickle
import re
import sys
from datetime import datetime
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]  # 05-项目源码
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# 必须先加载 .env，再读 WYU_MODEL_PATH / 阈值；否则会用默认模型去联网。
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass

from models import Chunk, RetrievedChunk
from rag.bm25 import BM25Retriever, zh_tokenize, query_coverage
from rag.hybrid import merge_and_rerank, rerank_by_query_coverage

INDEX_DIR = ROOT / "data" / "index" / "wyu"
MODEL = os.environ.get("WYU_MODEL_PATH", "BAAI/bge-m3")
MAX_LEN = 512
BM25_MIN = float(os.environ.get("WYU_BM25_MIN", "1.0"))
DENSE_MIN = float(os.environ.get("WYU_DENSE_MIN", "0.65"))
COVERAGE_MIN = float(os.environ.get("WYU_COVERAGE_MIN", "0.5"))

OUT_OF_SCOPE_RULES = (
    (
        "out_of_scope_employment",
        ("就业", "三方协议", "档案", "报到证", "招聘", "就业推荐表"),
    ),
    (
        "out_of_scope_campus",
        (
            "校园卡", "宿舍", "食堂", "校车", "快递", "wifi", "无线网",
            "水电", "报修", "后勤", "图书馆", "空调",
        ),
    ),
    (
        "out_of_scope_personal",
        (
            "查成绩", "成绩查询", "个人成绩", "学籍状态", "审批进度",
            "办理进度", "我的排名",
        ),
    ),
    (
        "out_of_scope_commitment",
        ("通融", "一定批准", "能不能过", "帮我改成绩", "包过"),
    ),
)

IN_SCOPE_TERMS = (
    "补考", "缓考", "选课", "校历", "作息", "节课", "上课", "课表",
    "成绩复核", "成绩", "考试", "教室", "借用", "通识", "课程",
    "学分", "期末", "等级考试", "创新创业", "学籍", "转专业",
    "休学", "复学", "退学", "辅修", "推免", "保研", "交换生",
    "违纪处分", "申诉", "奖学金", "助学金", "困难认定", "资助",
    "勤工助学", "助学贷款", "贷款", "补助", "绿色通道", "奖助",
    "学费", "免修", "免听", "重修", "绩点", "第二学士学位", "微专业",
    "毕业", "学位",
)


def classify_question_scope(question: str) -> str:
    """把问题分成范围内或具体的范围外类别。"""
    normalized = re.sub(r"\s+", "", (question or "").lower())
    for kind, terms in OUT_OF_SCOPE_RULES:
        if any(term.lower() in normalized for term in terms):
            return kind
    if any(term.lower() in normalized for term in IN_SCOPE_TERMS):
        return "in_scope"
    return "out_of_scope_other"


def build_refusal_message(question: str) -> str:
    """根据问题类型返回有明确指向的拒答说明。"""
    kind = classify_question_scope(question)
    if kind == "out_of_scope_employment":
        return (
            "这个问题不在邑问通的服务范围（就业事务）。"
            "建议联系学校就业指导中心，或查看学校就业信息网的最新通知。"
        )
    if kind == "out_of_scope_campus":
        return (
            "这个问题不在邑问通的服务范围（校园生活事务）。"
            "建议联系学生工作部或后勤管理处，查看对应部门的服务指南。"
        )
    if kind == "out_of_scope_personal":
        return (
            "这个问题涉及个人业务查询，邑问通不能读取个人成绩或审批进度。"
            "请登录教学管理系统或学信网查询，必要时联系学院教务员。"
        )
    if kind == "out_of_scope_commitment":
        return (
            "邑问通不能承诺审批结果，也不能修改成绩或代替学校作出处理。"
            "请按学校公开规定办理，并以对应部门的正式答复为准。"
        )
    if kind == "in_scope":
        return (
            "这个问题属于学业或资助范围，但资料库里暂未收录足够依据，"
            "我不能给出确定结论。建议查看五邑大学教务处、学生工作部"
            "最新通知，或联系学院教务员、辅导员确认。"
        )
    return (
        "这个问题不在邑问通的服务范围。邑问通只处理五邑大学的学业、"
        "学籍和资助政策，请查看学校官网或联系对应部门。"
    )


_PUBLISHED_DATE_RE = re.compile(
    r"(?:发布时间|发布日期|更新时间|来源)"
    r"[：:][^。\n]{0,120}?((20\d{2})(?:[-年/.]\d{1,2})?)"
)


def stale_source_docs(
    hits: list[RetrievedChunk],
    current_year: int | None = None,
    all_chunks: list[Chunk] | None = None,
) -> list[str]:
    """按明确发布日期判断时效，不把正文里的其他年份当成发布时间。"""
    current_year = current_year or datetime.now().year
    threshold = current_year - 3
    source_texts: dict[str, list[str]] = {}
    for chunk in all_chunks or []:
        source_texts.setdefault(chunk.source_doc, []).append(chunk.text)
    stale: list[str] = []
    for hit in hits:
        texts = source_texts.get(hit.chunk.source_doc, [hit.chunk.text])
        published_years: list[int] = []
        for value in texts:
            published_years.extend(
                int(match.group(2))
                for match in _PUBLISHED_DATE_RE.finditer(value)
            )
        if published_years and max(published_years) <= threshold:
            if hit.chunk.source_doc not in stale:
                stale.append(hit.chunk.source_doc)
    return stale


def build_stale_warning(sources: list[str]) -> str:
    names = "、".join(dict.fromkeys(sources))
    return (
        f"资料时效提醒：检索到的资料可能过时（{names}），"
        "请以学校最新通知为准。"
    )


def is_confident_retrieval(
    best_bm25: float,
    best_dense: float,
    coverage: float,
    dense_available: bool,
) -> bool:
    """高置信检索直接通过，避免 LLM 判定器误拒有来源的问题。"""
    if dense_available and best_dense >= DENSE_MIN:
        return True
    return coverage >= COVERAGE_MIN and best_bm25 >= BM25_MIN


def looks_like_refusal_answer(text: str) -> bool:
    """识别模型虽然返回文字、但实际没有给出确定结论的情况。"""
    return any(
        phrase in (text or "")
        for phrase in (
            "无法确定",
            "没有写明",
            "资料中未提供",
            "资料中没有",
            "无法根据现有资料",
        )
    )


_AMOUNT_QUESTION_TERMS = (
    "多少钱", "多少元", "金额", "标准", "最高", "最低", "费用",
)
_AMOUNT_EVIDENCE_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:元|万元|%|块)")
_AMOUNT_TOPICS = (
    "生源地助学贷款", "国家奖学金", "国家励志奖学金", "国家助学金",
    "勤工助学", "助学贷款", "学费", "补助", "工资", "奖学金", "助学金", "贷款",
)


def has_amount_evidence(question: str, hits: list[RetrievedChunk]) -> bool:
    """金额类问题必须检索到与问题主题相匹配的明确金额。"""
    if not any(term in (question or "") for term in _AMOUNT_QUESTION_TERMS):
        return True
    topics = [term for term in _AMOUNT_TOPICS if term in (question or "")]
    for hit in hits:
        for sentence in re.split(r"[。；;\n]", hit.chunk.text):
            if not _AMOUNT_EVIDENCE_RE.search(sentence):
                continue
            if not topics or any(topic in sentence for topic in topics):
                return True
    return False


class HybridQA:
    def __init__(self, use_dense: bool | None = None) -> None:
        with open(INDEX_DIR / "chunks.pkl", "rb") as fh:
            self.chunks: list[Chunk] = pickle.load(fh)
        self.bm25 = BM25Retriever(self.chunks, tokenize=zh_tokenize)

        self.dense_available = False
        self.dense_error = ""
        self.llm_available = False
        self.last_hits: list[RetrievedChunk] = []
        self.last_bm25 = 0.0
        self.last_dense = 0.0
        self.last_coverage = 0.0
        self.last_verdict = ""
        self.last_response_type = "answer"
        self.last_stale_sources: list[str] = []

        if use_dense is None:
            use_dense = os.environ.get("WYU_USE_DENSE", "1") != "0"
        if use_dense:
            self._init_dense()
        self._init_llm()

    # ------------------------------------------------------------------
    # 稠密检索：模型/索引加载失败时自动降级，不拖垮 BM25
    # ------------------------------------------------------------------

    @staticmethod
    def _load_model(auto_tokenizer, auto_model):
        """先读本地缓存，避免评委环境因联网检查/下载而卡住。"""
        try:
            tok = auto_tokenizer.from_pretrained(
                MODEL, local_files_only=True
            )
            try:
                model = auto_model.from_pretrained(
                    MODEL,
                    local_files_only=True,
                    use_safetensors=True,
                )
            except Exception:
                # 旧快照可能只有 pytorch_model.bin。
                model = auto_model.from_pretrained(
                    MODEL,
                    local_files_only=True,
                    use_safetensors=False,
                )
            return tok, model
        except Exception as exc:
            # 默认不联网：加载失败就快速降级，避免 UI 卡在“加载模型”。
            # 确实需要下载模型时，显式设置 WYU_ALLOW_MODEL_DOWNLOAD=1。
            if os.environ.get("WYU_ALLOW_MODEL_DOWNLOAD", "0") != "1":
                raise RuntimeError(
                    "本地模型缓存不可用；如需联网下载，"
                    "请设置 WYU_ALLOW_MODEL_DOWNLOAD=1"
                ) from exc
            tok = auto_tokenizer.from_pretrained(MODEL)
            try:
                model = auto_model.from_pretrained(
                    MODEL, use_safetensors=True
                )
            except Exception:
                model = auto_model.from_pretrained(
                    MODEL, use_safetensors=False
                )
            return tok, model

    def _init_dense(self) -> None:
        try:
            import faiss
            import numpy as np
            import torch
            from transformers import AutoModel, AutoTokenizer

            self._faiss = faiss
            self._np = np
            self._torch = torch

            requested = os.environ.get("WYU_DEVICE", "auto").lower()
            if requested == "cpu":
                self._device = "cpu"
            elif requested == "cuda":
                self._device = "cuda"
            else:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"

            self._tok, self._mdl = self._load_model(
                AutoTokenizer, AutoModel
            )
            self._mdl = self._mdl.to(self._device).eval()
            if self._device == "cuda":
                self._mdl = self._mdl.half()

            # Windows 中文路径下 faiss.read_index 的 C++ 文件 IO 会报
            # “Illegal byte sequence”；用 Python 读字节再反序列化。
            index_bytes = np.frombuffer(
                (INDEX_DIR / "faiss.index").read_bytes(),
                dtype=np.uint8,
            )
            index = faiss.deserialize_index(index_bytes)
            if index.ntotal != len(self.chunks):
                raise ValueError(
                    f"Faiss 向量数 {index.ntotal} 与切片数 "
                    f"{len(self.chunks)} 不一致，请重建索引"
                )
            self._index = index
            self.dense_available = True
        except Exception as exc:
            self.dense_available = False
            self.dense_error = f"{type(exc).__name__}: {exc}"

    def _embed(self, texts: list[str]):
        if not self.dense_available:
            return None
        vecs = []
        for i in range(0, len(texts), 16):
            batch = texts[i:i + 16]
            inputs = self._tok(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            ).to(self._device)
            with self._torch.no_grad():
                out = self._mdl(**inputs)
            vec = self._torch.nn.functional.normalize(
                out.last_hidden_state[:, 0].float(), p=2, dim=1
            )
            vecs.append(vec.cpu().numpy())
        return self._np.vstack(vecs)

    # ------------------------------------------------------------------
    # 召回
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 3):
        bm25_hits = self.bm25.retrieve(query, top_k=5)
        bm25_hits = rerank_by_query_coverage(query, bm25_hits)
        best_bm25 = bm25_hits[0].score if bm25_hits else 0.0

        dense_hits: list[RetrievedChunk] = []
        best_dense = 0.0
        if self.dense_available:
            query_vec = self._embed([query])
            scores, ids = self._index.search(query_vec, 5)
            dense_hits = [
                RetrievedChunk(self.chunks[index], float(score))
                for score, index in zip(scores[0], ids[0])
                if index >= 0
            ]
            best_dense = dense_hits[0].score if dense_hits else 0.0
            dense_hits = rerank_by_query_coverage(query, dense_hits)

        if best_bm25 < BM25_MIN and (
            not self.dense_available or best_dense < DENSE_MIN
        ):
            self.last_hits = []
            self.last_bm25 = best_bm25
            self.last_dense = best_dense
            self.last_coverage = 0.0
            return [], best_bm25, best_dense

        if dense_hits:
            fused = merge_and_rerank([bm25_hits, dense_hits], top_k=top_k)
        else:
            fused = bm25_hits[:top_k]
        fused = rerank_by_query_coverage(query, fused)

        self.last_hits = fused
        self.last_bm25 = best_bm25
        self.last_dense = best_dense
        self.last_coverage = (
            query_coverage(query, fused[0].chunk.text) if fused else 0.0
        )
        return fused, best_bm25, best_dense

    # ------------------------------------------------------------------
    # 三态判定
    # ------------------------------------------------------------------

    def _init_llm(self) -> None:
        self.llm_available = bool(
            os.environ.get("DEEPSEEK_API_KEY")
        ) and os.environ.get("WYU_OFFLINE", "0") != "1"
        self._client_obj = None

    def _client(self):
        if self._client_obj is None:
            from openai import OpenAI

            self._client_obj = OpenAI(
                base_url=os.environ.get(
                    "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
                ),
                api_key=os.environ["DEEPSEEK_API_KEY"],
            )
        return self._client_obj

    def _heuristic_verdict(self, best_bm25: float, best_dense: float) -> str:
        if best_bm25 >= BM25_MIN or (
            self.dense_available and best_dense >= DENSE_MIN
        ):
            return "correct"
        return "ambiguous"

    def _offline_verdict(
        self, best_bm25: float, best_dense: float
    ) -> str:
        if is_confident_retrieval(
            best_bm25,
            best_dense,
            self.last_coverage,
            self.dense_available,
        ):
            return "correct"
        if self.last_coverage < COVERAGE_MIN:
            return "incorrect"
        return self._heuristic_verdict(best_bm25, best_dense)

    def judge(
        self,
        question: str,
        hits: list[RetrievedChunk],
        best_bm25: float | None = None,
        best_dense: float | None = None,
    ) -> str:
        if not hits:
            return "incorrect"

        best_bm25 = self.last_bm25 if best_bm25 is None else best_bm25
        best_dense = self.last_dense if best_dense is None else best_dense

        # 没有 API key / 强制离线：用检索分数 + 覆盖度做保守判定，
        # 并在 UI 标注“离线模式”。
        if not self.llm_available:
            return self._offline_verdict(best_bm25, best_dense)

        if is_confident_retrieval(
            best_bm25,
            best_dense,
            self.last_coverage,
            self.dense_available,
        ):
            return "correct"

        context = "\n".join(
            f"[{hit.chunk.source_doc}] {hit.chunk.text[:300]}"
            for hit in hits
        )
        prompt = (
            "判断下面的资料能否回答这个问题，只输出一个词："
            "correct / ambiguous / incorrect。\n\n"
            f"问题：{question}\n"
            f"检索覆盖度：{self.last_coverage:.2f}\n\n资料：\n{context}"
        )
        try:
            response = self._client().chat.completions.create(
                model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            text = (response.choices[0].message.content or "").strip().lower()
            for verdict in ("incorrect", "ambiguous", "correct"):
                if verdict in text:
                    return verdict
            return "incorrect"
        except Exception:
            # 模型故障时退回到检索分数 + 覆盖度判定，保证会话不崩。
            return self._offline_verdict(best_bm25, best_dense)

    # ------------------------------------------------------------------
    # 回答
    # ------------------------------------------------------------------

    def _extractive_answer(self, hits: list[RetrievedChunk]) -> str:
        lines = [
            "（离线/降级模式：未调用大模型，以下为检索到的官方资料原文摘录）"
        ]
        for hit in hits[:5]:
            text = hit.chunk.text.strip()
            if len(text) > 500:
                text = text[:500] + "…"
            lines.append(f"【{hit.chunk.source_doc}】{text}")
        lines.append(
            "来源：" + "、".join(
                dict.fromkeys(hit.chunk.source_doc for hit in hits[:5])
            )
        )
        return "\n\n".join(lines)

    def answer(self, question: str):
        scope = classify_question_scope(question)
        self.last_response_type = "answer"
        self.last_stale_sources = []
        if scope.startswith("out_of_scope"):
            self.last_hits = []
            self.last_bm25 = 0.0
            self.last_dense = 0.0
            self.last_coverage = 0.0
            self.last_verdict = "incorrect"
            self.last_response_type = "out_of_scope"
            return build_refusal_message(question), 0.0, 0.0

        hits, best_bm25, best_dense = self.retrieve(question, top_k=5)
        # 生成使用前 5 条上下文；界面和评测仍只展示前 3 条来源。
        self.last_hits = hits[:3]
        self.last_verdict = (
            self.judge(question, hits, best_bm25, best_dense)
            if hits
            else "incorrect"
        )

        if not hits or self.last_verdict != "correct":
            self.last_response_type = "missing_source"
            return build_refusal_message(question), best_bm25, best_dense

        if not has_amount_evidence(question, hits):
            self.last_verdict = "incorrect"
            self.last_response_type = "missing_source"
            return build_refusal_message(question), best_bm25, best_dense

        self.last_stale_sources = stale_source_docs(hits, all_chunks=self.chunks)
        stale_note = ""
        if self.last_stale_sources:
            self.last_response_type = "stale_warning"
            stale_note = "\n\n" + build_stale_warning(self.last_stale_sources)

        if not self.llm_available:
            return (
                self._extractive_answer(hits) + stale_note,
                best_bm25,
                best_dense,
            )

        context = "\n\n".join(
            f"[{hit.chunk.source_doc}] {hit.chunk.text}"
            for hit in hits
        )
        prompt = (
            "你是五邑大学的学业助手。只根据下面的资料回答，"
            "答案里标注来源文件名；资料里没有的信息不要编。"
            "条件类问题必须逐条保留原文的条件和限制，不能合并或省略。\n\n"
            f"问题：{question}\n\n资料：\n{context}"
        )
        try:
            response = self._client().chat.completions.create(
                model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            content = response.choices[0].message.content or ""
            if looks_like_refusal_answer(content):
                self.last_response_type = "missing_source"
            return content + stale_note, best_bm25, best_dense
        except Exception as exc:
            fallback = self._extractive_answer(hits)
            note = f"\n\n（生成失败，已降级为原文摘录：{type(exc).__name__}）"
            return fallback + stale_note + note, best_bm25, best_dense

    @property
    def mode(self) -> str:
        retrieval = "混合召回" if self.dense_available else "BM25-only"
        generation = "LLM 生成" if self.llm_available else "离线原文摘录"
        return f"{retrieval} / {generation}"


def main() -> None:
    qa = HybridQA()
    print(f"模式：{qa.mode}")
    print(f"知识库：{len(qa.chunks)} 个切片")
    if qa.dense_error:
        print(f"稠密检索降级原因：{qa.dense_error}")
    for question in [
        "补考怎么报名",
        "第一节课几点开始",
        "食堂几点开门",
    ]:
        print(f"\n=== {question} ===")
        answer, bm25, dense = qa.answer(question)
        print(
            f"[bm25={bm25:.2f} dense={dense:.3f} "
            f"coverage={qa.last_coverage:.2f} "
            f"verdict={qa.last_verdict} "
            f"type={qa.last_response_type}]"
        )
        print(answer)


if __name__ == "__main__":
    main()



