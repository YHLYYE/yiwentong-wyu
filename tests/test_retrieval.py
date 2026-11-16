"""检索层回归测试：中文空格、BM25 排序、RRF 去重。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models import Chunk, RetrievedChunk
from rag.bm25 import (
    BM25Retriever,
    query_coverage,
    zh_tokenize,
)
from rag.hybrid import merge_and_rerank


def test_zh_tokenize_collapses_spaces_between_chinese():
    tokens = zh_tokenize("第 一 节：8:15 ～ 9:00")
    assert "第一" in tokens
    assert "一节" in tokens


def test_bm25_prefers_short_relevant_document():
    chunks = [
        Chunk(
            chunk_id="a",
            text="五邑大学学生作息时间表 第 一 节：8:15～9:00 第 二 节：9:05～9:50",
            source_doc="作息时间",
        ),
        Chunk(
            chunk_id="b",
            text=(
                "关于开展精彩课堂评选活动的通知，第一学期开始，"
                "教师参评条件、材料提交和评审安排……"
            ),
            source_doc="精彩课堂评选通知",
        ),
    ]
    hits = BM25Retriever(chunks).retrieve("第一节课几点开始", top_k=2)
    assert hits
    assert hits[0].chunk.source_doc == "作息时间"


def test_query_coverage_rejects_unrelated_questions():
    assert query_coverage("补考怎么报名", "补考报名步骤：登录系统……") == 1.0
    assert (
        query_coverage("校园卡丢了怎么补办", "考生须带身份证和学生证/校园卡")
        < 0.6
    )
    assert query_coverage("国家奖学金怎么申请", "缓考申请流程") < 0.6


def test_query_coverage_keeps_relevant_schedule_question():
    assert query_coverage("第一节课几点开始", "第 一 节：8:15～9:00") >= 0.6


def test_rrf_merges_and_deduplicates():
    a = Chunk("a", "a", "A")
    b = Chunk("b", "b", "B")
    c = Chunk("c", "c", "C")
    fused = merge_and_rerank(
        [
            [RetrievedChunk(a, 10.0), RetrievedChunk(b, 9.0)],
            [RetrievedChunk(b, 0.9), RetrievedChunk(c, 0.8)],
        ],
        top_k=3,
    )
    ids = [item.chunk.chunk_id for item in fused]
    assert len(ids) == len(set(ids)) == 3
    assert ids[0] == "b"  # 两路都出现，RRF 分数最高
