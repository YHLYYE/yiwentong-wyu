"""五邑大学知识库 20 题评测（检索 + 兜底/拒答），结果导出 CSV。

用法：
    python src/eval_wyu.py
    python src/eval_wyu.py --output C:/tmp/eval.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
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

from wyu_qa import HybridQA

# 20 题仅供快速回归；正式评测用 scripts/eval_50_llm.py。
DEFAULT_OUT = ROOT.parent / "04-测试数据" / "20题评测结果-重跑.csv"

ANSWERABLE = [
    ("补考怎么报名", "补考报名通知"),
    ("第一节课几点开始", "作息时间"),
    ("第十节课几点结束", "作息时间"),
    ("缓考怎么申请", "缓考申请"),
    ("怎么撤销补考报名", "补考报名通知"),
    ("2026-2027学年校历在哪看", "五邑大学2026-2027学年校历"),
    ("通识课选课范围是什么", "通识教育选课说明"),
    ("计算机等级考试成绩怎么复核", "计算机等级考试成绩复核"),
    ("借用教室需要什么手续", "借用教室"),
    ("国家奖学金怎么申请", "国家助学奖学金实施办法"),
    ("转专业需要什么条件", "转专业"),
]

# 这些属于“学业/资助”范围，但当前索引里没有对应资料；
# 期望行为是“范围内无资料”的兜底，而不是“超范围拒答”。
IN_SCOPE_UNANSWERABLE = [
    "辅修学士学位怎么申请",
]

# 明确超出“学业+资助”范围。
OUT_OF_SCOPE = [
    "食堂几点开门",
    "图书馆 wifi 密码是多少",
    "宿舍空调坏了找谁",
    "三方协议怎么签",
    "校园卡丢了怎么补办",
    "学校有多少个学院",
    "校车时刻表在哪",
    "快递点在哪里",
]


def answer_preview(text: str, limit: int = 120) -> str:
    return " ".join((text or "").split())[:limit]


def is_refusal(text: str) -> bool:
    return any(
        phrase in (text or "")
        for phrase in (
            "不在邑问通的服务范围",
            "暂未收录足够依据",
            "无法确定",
            "没有写明",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out_path = Path(args.output)

    qa = HybridQA()
    print(f"模式：{qa.mode}")
    print(f"知识库：{len(qa.chunks)} 个切片")

    rows = []

    hit = 0
    for question, expected in ANSWERABLE:
        hits, bm25, dense = qa.retrieve(question, top_k=3)
        docs = [hit.chunk.source_doc for hit in hits]
        ok = any(expected in doc for doc in docs)
        hit += ok
        answer, _, _ = qa.answer(question)
        rows.append([
            "可回答", question, expected, " | ".join(docs),
            f"{bm25:.2f}", f"{dense:.3f}",
            f"{qa.last_coverage:.2f}", qa.last_verdict,
            "检索命中" if ok else "检索未命中",
            answer_preview(answer),
            " | ".join(docs),
            qa.mode,
        ])
        print(f"{'OK ' if ok else 'MISS'} {question} -> {docs}")

    in_scope_ok = 0
    for question in IN_SCOPE_UNANSWERABLE:
        hits, bm25, dense = qa.retrieve(question, top_k=3)
        docs = [hit.chunk.source_doc for hit in hits]
        answer, _, _ = qa.answer(question)
        ok = is_refusal(answer) or qa.last_verdict != "correct"
        in_scope_ok += ok
        rows.append([
            "范围内无资料", question, "", " | ".join(docs),
            f"{bm25:.2f}", f"{dense:.3f}",
            f"{qa.last_coverage:.2f}", qa.last_verdict,
            "正确兜底" if ok else "误答",
            answer_preview(answer),
            " | ".join(docs),
            qa.mode,
        ])
        print(f"{'FALLBACK' if ok else 'WRONG '} {question} -> {qa.last_verdict}")

    out_ok = 0
    for question in OUT_OF_SCOPE:
        hits, bm25, dense = qa.retrieve(question, top_k=3)
        docs = [hit.chunk.source_doc for hit in hits]
        answer, _, _ = qa.answer(question)
        ok = is_refusal(answer) or qa.last_verdict != "correct"
        out_ok += ok
        rows.append([
            "范围外", question, "", " | ".join(docs),
            f"{bm25:.2f}", f"{dense:.3f}",
            f"{qa.last_coverage:.2f}", qa.last_verdict,
            "拒答" if ok else "漏答",
            answer_preview(answer),
            " | ".join(docs),
            qa.mode,
        ])
        print(f"{'REJECT' if ok else 'LEAK  '} {question} -> {qa.last_verdict}")

    n1 = len(ANSWERABLE)
    n2 = len(IN_SCOPE_UNANSWERABLE)
    n3 = len(OUT_OF_SCOPE)
    print(f"\n检索 top-3：{hit}/{n1} = {hit / n1:.0%}")
    print(f"范围内无资料兜底：{in_scope_ok}/{n2} = {in_scope_ok / n2:.0%}")
    print(f"范围外拒答：{out_ok}/{n3} = {out_ok / n3:.0%}")
    print("注意：检索命中 ≠ 答案正确；答案正确率需要人工核对 CSV 的“答案摘要”。")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "类型", "问题", "期望来源", "检索Top3",
            "BM25最高分", "向量最高分", "覆盖度",
            "LLM判定", "结果", "答案摘要", "引用", "模式",
        ])
        writer.writerows(rows)
    print(f"CSV: {out_path}")


if __name__ == "__main__":
    main()
