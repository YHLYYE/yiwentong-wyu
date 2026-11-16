"""检索层回归测试：中文空格、BM25 排序、RRF 去重。"""
from __future__ import annotations

import pickle
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
from rag.hybrid import merge_and_rerank, rerank_by_query_coverage


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

def test_current_credit_rule_ranks_first_for_credit_limit_question():
    root = Path(__file__).resolve().parents[1]
    with open(root / "data" / "index" / "wyu" / "chunks.pkl", "rb") as fh:
        chunks = pickle.load(fh)
    question = "每学期最多能选多少学分"
    hits = BM25Retriever(chunks).retrieve(question, top_k=5)
    ranked = rerank_by_query_coverage(question, hits)
    assert ranked
    assert ranked[0].chunk.source_doc == "学分与课程修读规定2024"


def test_coverage_rerank_keeps_exact_topic_above_generic_chunk():
    question = "缓考怎么申请"
    generic = RetrievedChunk(
        Chunk("a", "学分、课程、申请、考试等通用说明。", "学分规定"),
        20.0,
    )
    exact = RetrievedChunk(
        Chunk("b", "缓考怎么申请：在考试前2日提交缓考申请及证明材料。", "缓考"),
        8.0,
    )
    ranked = rerank_by_query_coverage(question, [generic, exact])
    assert ranked[0].chunk.source_doc == "缓考"


def test_query_coverage_ignores_natural_question_phrases():
    assert query_coverage(
        "补考和缓考什么时候考试",
        "补考与缓考考试同时进行，于8月27日至8月30日完成。",
    ) >= 0.6
    assert query_coverage(
        "哪些情况会被退学",
        "学校可予退学处理的情形包括未完成学业、未复学等。",
    ) >= 0.6
