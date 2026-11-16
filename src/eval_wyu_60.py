"""Run the 60-question regression set against the current knowledge base."""
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


def expected_source_hit(expected: str, docs: list[str]) -> bool:
    if not expected:
        return False
    options = [item.strip() for item in expected.split("/") if item.strip()]
    return any(option in doc for option in options for doc in docs)


def answer_preview(text: str, limit: int = 320) -> str:
    return " ".join((text or "").split())[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT.parent / "04-测试数据" / "60题测试集.csv"))
    parser.add_argument("--output", default=str(ROOT.parent / "04-测试数据" / "60题测试结果-当前索引.csv"))
    parser.add_argument("--bm25-only", action="store_true")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8-sig", newline="") as fh:
        questions = [normalize_row(row) for row in csv.DictReader(fh)]
    qa = HybridQA(use_dense=not args.bm25_only)
    print(f"模式：{qa.mode}")
    print(f"知识库：{len(qa.chunks)} 个切片")

    rows = []
    counts = {"可回答": [0, 0], "范围内无资料": [0, 0], "范围外": [0, 0]}
    for item in questions:
        question = item["问题"]
        expected_type = item["类型"]
        expected = item.get("期望来源", "")
        answer, bm25, dense = qa.answer(question)
        docs = [hit.chunk.source_doc for hit in qa.last_hits]
        source_hit = expected_source_hit(expected, docs)

        if expected_type == "可回答":
            ok = qa.last_response_type in {"answer", "stale_warning"} and source_hit
            check = "来源命中" if ok else ("来源命中但未回答" if source_hit else "未命中或错误")
        elif expected_type == "范围内无资料":
            ok = qa.last_response_type == "missing_source"
            check = "兜底符合预期" if ok else "未正确兜底"
        else:
            ok = qa.last_response_type == "out_of_scope"
            check = "拒答符合预期" if ok else "未拒答"

        counts.setdefault(expected_type, [0, 0])
        counts[expected_type][1] += 1
        counts[expected_type][0] += int(ok)
        rows.append({
            "序号": item["序号"],
            "类型": expected_type,
            "问题": question,
            "期望来源": expected,
            "关键点": item.get("关键点", ""),
            "检索Top3": " | ".join(docs),
            "响应类型": qa.last_response_type,
            "BM25最高分": f"{bm25:.2f}",
            "向量最高分": f"{dense:.3f}",
            "覆盖度": f"{qa.last_coverage:.2f}",
            "系统判定": qa.last_verdict,
            "自动检查": check,
            "答案摘要": answer_preview(answer),
            "引用": " | ".join(docs),
            "模式": qa.mode,
        })
        print(f"{'OK' if ok else 'FAIL':4} {question} -> {check}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\n汇总")
    for kind, (ok, total) in counts.items():
        print(f"{kind}: {ok}/{total}")
    print(f"输出：{out}")


if __name__ == "__main__":
    main()
