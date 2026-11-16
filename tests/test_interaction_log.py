"""辅导员工作台日志/反馈/导出的回归测试。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interaction_log import InteractionLog


def log_event(log: InteractionLog, question: str, verdict: str):
    return log.log(
        question=question,
        answer="answer",
        verdict=verdict,
        bm25=9.0,
        dense=0.0,
        coverage=1.0,
        sources=["补考报名通知"],
        mode="BM25-only / 离线原文摘录",
    )


def test_log_summary_and_feedback(tmp_path):
    log = InteractionLog(tmp_path / "qa_log.jsonl")
    first = log_event(log, "补考怎么报名", "correct")
    second = log_event(log, "食堂几点开门", "incorrect")

    summary = log.summary()
    assert summary["total"] == 2
    assert summary["answered"] == 1
    assert summary["refused"] == 1
    assert summary["top_questions"][0] == ("补考怎么报名", 1)

    assert log.feedback(first["id"], "有用") is True
    assert log.feedback(second["id"], "需要人工", escalated=True) is True
    assert log.update("missing", feedback="有用") is False

    summary = log.summary()
    assert summary["positive"] == 1
    assert summary["escalated"] == 1
    assert len(summary["gaps"]) == 1


def test_corrupt_lines_are_skipped(tmp_path):
    path = tmp_path / "qa_log.jsonl"
    path.write_text(
        '{"id": "1", "question": "补考怎么报名"}\n'
        "not a json line\n",
        encoding="utf-8",
    )
    log = InteractionLog(path)
    events = log.read_all()
    assert len(events) == 1
    assert events[0]["id"] == "1"


def test_csv_export(tmp_path):
    log = InteractionLog(tmp_path / "qa_log.jsonl")
    log_event(log, "补考怎么报名", "correct")
    output = log.export_csv(tmp_path / "export.csv")
    assert output.exists()
    text = output.read_text(encoding="utf-8-sig")
    assert "补考怎么报名" in text
    assert "sources" in text
