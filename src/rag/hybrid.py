"""RRF（Reciprocal Rank Fusion）融合多路召回结果。"""
from __future__ import annotations

from models import RetrievedChunk
from rag.bm25 import query_coverage


def merge_and_rerank(
    result_lists: list[list[RetrievedChunk]],
    top_k: int = 3,
    k: int = 60,
) -> list[RetrievedChunk]:
    scores: dict[str, float] = {}
    chunks = {}

    for results in result_lists:
        for rank, item in enumerate(results, 1):
            chunk_id = item.chunk.chunk_id
            chunks[chunk_id] = item.chunk
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [
        RetrievedChunk(chunks[chunk_id], score)
        for chunk_id, score in ordered[:top_k]
    ]


def rerank_by_query_coverage(
    query: str,
    results: list[RetrievedChunk],
    coverage_weight: float = 0.7,
) -> list[RetrievedChunk]:
    """用问题词覆盖度重排单路召回，避免通用词多的长文压过精准短答。"""
    if len(results) < 2:
        return results
    max_score = max((item.score for item in results), default=0.0)
    if max_score <= 0:
        return results
    coverage_weight = min(max(coverage_weight, 0.0), 1.0)
    score_weight = 1.0 - coverage_weight
    ranked = []
    for index, item in enumerate(results):
        coverage = query_coverage(query, item.chunk.text)
        combined = coverage * coverage_weight + (item.score / max_score) * score_weight
        ranked.append((combined, -index, item))
    ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return [item for _, _, item in ranked]
