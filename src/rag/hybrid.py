"""RRF（Reciprocal Rank Fusion）融合多路召回结果。"""
from __future__ import annotations

from models import RetrievedChunk


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
