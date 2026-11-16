"""拒答分类与资料时效提醒的行为测试。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models import Chunk, RetrievedChunk
from wyu_qa import (
    build_refusal_message,
    build_stale_warning,
    classify_question_scope,
    has_amount_evidence,
    is_confident_retrieval,
    looks_like_refusal_answer,
    stale_source_docs,
)


def test_scope_classification_distinguishes_campus_and_academic():
    assert classify_question_scope("食堂几点开门") == "out_of_scope_campus"
    assert classify_question_scope("三方协议怎么签") == "out_of_scope_employment"
    assert classify_question_scope("查一下我的审批进度") == "out_of_scope_personal"
    assert classify_question_scope("能不能通融一下") == "out_of_scope_commitment"
    assert classify_question_scope("辅修学士学位怎么申请") == "in_scope"
    assert classify_question_scope("第一节课几点开始") == "in_scope"
    assert classify_question_scope("通识核心课程要修多少学分") == "in_scope"
    assert classify_question_scope("计算机等级考试成绩怎么复核") == "in_scope"
    assert classify_question_scope("借用教室需要什么手续") == "in_scope"
    assert classify_question_scope("学校有几个学院") == "out_of_scope_other"


def test_refusal_messages_point_to_the_right_destination():
    assert "学生工作部" in build_refusal_message("食堂几点开门")
    assert "就业指导中心" in build_refusal_message("三方协议怎么签")
    assert "教学管理系统" in build_refusal_message("查一下我的审批进度")
    assert "暂未收录" in build_refusal_message("辅修学士学位怎么申请")
    assert "不能承诺" in build_refusal_message("能不能通融一下")


def test_stale_source_detection_uses_publication_year():
    hits = [
        RetrievedChunk(
            Chunk("a", "发布时间：2021年，旧版说明。", "教务答疑"),
            1.0,
        ),
        RetrievedChunk(
            Chunk("b", "2024年发布的现行规定。", "转专业"),
            0.9,
        ),
    ]
    assert stale_source_docs(hits, current_year=2026) == ["教务答疑"]
    assert "可能过时" in build_stale_warning(["教务答疑"])
    assert "教务答疑" in build_stale_warning(["教务答疑"])


def test_high_confidence_retrieval_bypasses_llm_judge():
    assert is_confident_retrieval(13.70, 0.726, 0.38, True)
    assert is_confident_retrieval(16.66, 0.0, 0.80, False)
    assert not is_confident_retrieval(0.50, 0.40, 0.50, True)


def test_stale_detection_uses_whole_source_not_one_chunk():
    source_chunks = [
        Chunk("x1", "发布时间：2024-10-08。", "现行办法"),
        Chunk("x2", "本办法自2020年9月起执行。", "现行办法"),
    ]
    hits = [RetrievedChunk(source_chunks[1], 1.0)]
    assert stale_source_docs(hits, current_year=2026, all_chunks=source_chunks) == []


def test_stale_detection_uses_explicit_publication_date():
    source_chunks = [
        Chunk(
            "old1",
            "教务答疑中的学籍相关问答（2021 版）。来源：教务处，2021-05-08。重要：2024 版已替代。",
            "教务答疑",
        )
    ]
    hits = [RetrievedChunk(source_chunks[0], 1.0)]
    assert stale_source_docs(hits, current_year=2026, all_chunks=source_chunks) == ["教务答疑"]


def test_answer_self_refusal_is_detected():
    assert looks_like_refusal_answer("资料中没有写明具体金额，因此无法确定。")
    assert not looks_like_refusal_answer("国家奖学金标准为每人每年8000元。")


def test_amount_question_requires_amount_evidence():
    no_amount = [
        RetrievedChunk(Chunk("a", "生源地贷款按上级规定执行。", "奖助项目概览"), 1.0),
        RetrievedChunk(Chunk("b", "国家奖学金每年 8000 元。", "奖励办法"), 0.9),
    ]
    with_amount = [RetrievedChunk(Chunk("c", "国家助学金每年 2000-4000 元。", "资助办法"), 1.0)]
    assert not has_amount_evidence("生源地助学贷款最高多少", no_amount)
    assert has_amount_evidence("国家助学金一年多少钱", with_amount)


def test_amount_evidence_must_be_near_question_topic():
    mixed = [
        RetrievedChunk(
            Chunk("d", "助学贷款按上级规定执行。勤工助学每小时 15.3 元。", "资助细则"),
            1.0,
        )
    ]
    assert not has_amount_evidence("生源地助学贷款最高多少", mixed)

def test_expanded_aid_terms_are_in_scope():
    assert classify_question_scope("临时困难补助是什么") == "in_scope"
    assert classify_question_scope("绿色通道怎么申请") == "in_scope"
    assert classify_question_scope("国家助学贷款怎么申请") == "in_scope"

def test_definition_question_about_subsidy_does_not_require_amount():
    hits = [
        RetrievedChunk(
            Chunk("x", "临时困难补助用于突发情况或特殊事件导致家庭经济困难的学生。", "奖助项目概览"),
            1.0,
        )
    ]
    assert has_amount_evidence("临时困难补助是什么", hits)
