"""为五邑大学公开资料建 BM25 索引（中文按 400 字切块，50 字重叠）。

用法：
    python src/build_wyu_index.py --dry-run   # 只统计，不覆盖现有索引
    python src/build_wyu_index.py             # 重建 chunks.pkl
重建 chunks.pkl 后必须再跑 build_wyu_dense.py，否则 Faiss 索引会与切片不一致。
"""
from __future__ import annotations

import argparse
import hashlib
import pickle
import re
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

from models import Chunk
from rag.bm25 import BM25Retriever, zh_tokenize

DATA = ROOT / "data" / "wyu"
OUT = ROOT / "data" / "index" / "wyu"
CHUNK_SIZE = 400
OVERLAP = 50


def split_text(
    text: str,
    size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> list[str]:
    step = max(1, size - overlap)
    out = []
    for i in range(0, max(1, len(text) - overlap), step):
        seg = text[i:i + size].strip()
        if len(seg) >= 50:
            out.append(seg)
    return out


_BOILERPLATE = [
    "邑大首页 首页 机构设置 规章制度 专业建设 办事指南 资料下载 信息查询 本科招生 全面育人",
    "部门首页 部门概况 思想教育 学风建设 队伍建设 资助工作 学生社区 就业工作 心理工作 规章制度",
    "当前位置: 首页 >>",
    "当前位置：首页 >>",
    "浏览次数:",
    "【关闭】 -->",
]


def clean_text(text: str) -> str:
    """去掉官网导航/页脚，避免切片里全是“首页/机构设置/联系我们”。"""
    text = text.replace("﻿", "")
    for marker in _BOILERPLATE:
        text = text.replace(marker, " ")
    text = re.split(r"联系我们\s*联系电话", text)[0]
    text = re.split(r"CopyRight", text)[0]
    text = re.sub(r"上一条：.*?(?=下一条：|$)", " ", text)
    text = re.sub(r"下一条：.*?$", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=OVERLAP)
    args = parser.parse_args()

    files = sorted(DATA.glob("*.txt"))
    if not files:
        raise SystemExit(f"没有找到资料：{DATA}")

    chunks: list[Chunk] = []
    for file in files:
        text = clean_text(file.read_text(encoding="utf-8-sig"))
        for index, segment in enumerate(
            split_text(text, args.chunk_size, args.overlap)
        ):
            cid = hashlib.sha1(f"{file.stem}|{index}".encode()).hexdigest()[:16]
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    text=segment,
                    source_doc=file.stem,
                    section="",
                    page=0,
                )
            )

    print(f"资料：{len(files)} 份，切片：{len(chunks)} 个")
    if args.dry_run:
        print("dry-run：不写入文件")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "chunks.pkl", "wb") as fh:
        pickle.dump(chunks, fh)
    print(f"已写入：{OUT / 'chunks.pkl'}")
    print("提醒：chunks.pkl 已变化，请运行 build_wyu_dense.py 重建 Faiss 索引。")

    bm25 = BM25Retriever(chunks, tokenize=zh_tokenize)
    for question in ["补考怎么报名", "第一节课几点开始", "缓考怎么申请"]:
        print(f"\nQ: {question}")
        for hit in bm25.retrieve(question, top_k=3):
            print(
                f"  {hit.score:.2f}  {hit.chunk.source_doc}  "
                f"{hit.chunk.text[:60]}"
            )


if __name__ == "__main__":
    main()
