"""Run the five-scenario, 25-question walkthrough on the current index."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wyu_qa import HybridQA


def normalize_row(row: dict[str, str]) -> dict[str, str]:
    return {key.lstrip("\ufeff"): value for key, value in row.items()}


def expected_hit(expected: str, docs: list[str]) -> bool:
    options = [item.strip() for item in expected.split("/") if item.strip()]
    return any(option in doc for option in options for doc in docs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT.parent / "04-测试数据" / "场景走查-5场景25题.csv"))
    parser.add_argument("--output", default=str(ROOT.parent / "04-测试数据" / "场景走查-5场景25题.csv"))
    parser.add_argument("--bm25-only", action="store_true")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8-sig", newline="") as fh:
        rows = [normalize_row(row) for row in csv.DictReader(fh)]
        fields = list(rows[0])
    qa = HybridQA(use_dense=not args.bm25_only)
    print(f"模式：{qa.mode}")
    print(f"知识库：{len(qa.chunks)} 个切片")

    for row in rows:
        question = row["问题"]
        expected_type = row["预期类型"]
        answer, bm25, dense = qa.answer(question)
        docs = [hit.chunk.source_doc for hit in qa.last_hits]
        row["系统判定"] = qa.last_verdict
        row["检索Top3"] = " | ".join(docs) if docs else "无（未达到检索门槛）"
        row["BM25最高分"] = f"{bm25:.2f}"
        row["向量最高分"] = f"{dense:.3f}"
        row["覆盖度"] = f"{qa.last_coverage:.2f}"
        row["模式"] = qa.mode
        if expected_type == "可回答":
            ok = expected_hit(row.get("期望来源", ""), docs)
            row["核对结果"] = "来源命中，待人工复核" if ok else "需复核"
        elif expected_type == "范围内无资料":
            ok = qa.last_response_type == "missing_source"
            row["核对结果"] = "兜底符合预期" if ok else "需复核"
        else:
            ok = qa.last_response_type == "out_of_scope"
            row["核对结果"] = "拒答符合预期" if ok else "需复核"
        row["答案摘要"] = " ".join(answer.split())[:320]
        print(f"{'OK' if ok else 'FAIL':4} {question} -> {row['核对结果']}")

    out = Path(args.output)
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"输出：{out}")


if __name__ == "__main__":
    main()
