"""问答日志、反馈与辅导员工作台统计。

日志为 JSONL，一行一个事件；只记录问题/答案/检索信息/反馈，
不要记录姓名、学号、身份证等个人敏感信息。
"""
from __future__ import annotations

import csv
import json
import os
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class InteractionLog:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def log(
        self,
        *,
        question: str,
        answer: str,
        verdict: str,
        bm25: float,
        dense: float,
        coverage: float,
        sources: list[str],
        mode: str,
        kind: str = "student",
        feedback: str = "",
        escalated: bool = False,
    ) -> dict[str, Any]:
        event = {
            "id": uuid.uuid4().hex[:12],
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "question": question,
            "answer": answer,
            "verdict": verdict,
            "bm25": round(float(bm25), 4),
            "dense": round(float(dense), 4),
            "coverage": round(float(coverage), 4),
            "sources": list(sources),
            "mode": mode,
            "feedback": feedback,
            "escalated": bool(escalated),
        }
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(event, ensure_ascii=False) + "\n"
                )
        return event

    # ------------------------------------------------------------------
    # 读取 / 更新
    # ------------------------------------------------------------------

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self._lock:
            return self._read_locked()

    def _read_locked(self) -> list[dict[str, Any]]:
        events = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
        return events

    def update(self, event_id: str, **fields: Any) -> bool:
        with self._lock:
            events = self._read_locked()
            found = False
            for event in events:
                if event.get("id") == event_id:
                    event.update(fields)
                    found = True
            if found:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                temp = self.path.with_suffix(self.path.suffix + ".tmp")
                with temp.open("w", encoding="utf-8") as fh:
                    for event in events:
                        fh.write(
                            json.dumps(event, ensure_ascii=False) + "\n"
                        )
                os.replace(temp, self.path)
            return found

    def feedback(
        self,
        event_id: str,
        feedback: str,
        *,
        escalated: bool | None = None,
    ) -> bool:
        fields: dict[str, Any] = {"feedback": feedback}
        if escalated is not None:
            fields["escalated"] = escalated
        return self.update(event_id, **fields)

    # ------------------------------------------------------------------
    # 统计 / 导出
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        events = self.read_all()
        total = len(events)
        answered = sum(
            1 for event in events if event.get("verdict") == "correct"
        )
        refused = total - answered
        positive = sum(
            1 for event in events if event.get("feedback") == "有用"
        )
        negative = sum(
            1 for event in events if event.get("feedback") == "没用"
        )
        escalated = sum(
            1 for event in events if event.get("escalated")
        )
        top_questions = Counter(
            event.get("question", "") for event in events
        ).most_common(5)
        gaps = [
            event
            for event in events
            if event.get("verdict") != "correct"
            or event.get("feedback") == "没用"
        ]
        return {
            "total": total,
            "answered": answered,
            "refused": refused,
            "positive": positive,
            "negative": negative,
            "escalated": escalated,
            "top_questions": top_questions,
            "gaps": gaps[-20:],
            "recent": events[-20:],
        }

    @staticmethod
    def to_csv(events: list[dict[str, Any]]) -> str:
        fields = [
            "id", "ts", "kind", "question", "verdict", "feedback",
            "escalated", "bm25", "dense", "coverage", "sources",
            "mode", "answer",
        ]
        import io

        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for event in events:
            row = dict(event)
            row["sources"] = " | ".join(event.get("sources", []))
            writer.writerow(row)
        return buffer.getvalue()

    def export_csv(self, path: str | os.PathLike[str]) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            self.to_csv(self.read_all()),
            encoding="utf-8-sig",
        )
        return target
