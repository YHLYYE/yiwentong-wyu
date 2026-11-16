"""纯 Python BM25，中文 bigram + 英文/数字词，方便在无第三方依赖时复现。"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable

from models import Chunk, RetrievedChunk

_CJK_RE = re.compile(r"[一-鿿]+")
_ASCII_RE = re.compile(r"[a-zA-Z0-9_]+")
# 作息表等页面会出现“第 一 节”这种中文间空格；先合并，避免 bigram 切碎。
_CJK_SPACE_RE = re.compile(r"(?<=[一-鿿])\s+(?=[一-鿿])")


def zh_tokenize(text: str) -> list[str]:
    """中文按二元组切分，英文/数字按词切分。"""
    text = _CJK_SPACE_RE.sub("", text)
    lowered = text.lower()
    tokens = _ASCII_RE.findall(lowered)
    for run in _CJK_RE.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


# 用于离线相关性门控：这些是问句/虚词，不代表“问题主题”。
_QUESTION_PHRASES = [
    "什么时候", "什么情况", "哪些情况", "为什么", "是什么", "怎么样",
    "请问", "帮我", "一下", "怎么", "如何", "什么", "时候",
    "是否", "可以", "需要", "几点", "多少", "哪里", "在哪",
    "哪个", "哪些", "会被", "有哪些", "开始", "结束", "看",
    "吗", "呢", "的", "了", "和",
]


def strip_question_phrases(text: str) -> str:
    for phrase in sorted(_QUESTION_PHRASES, key=len, reverse=True):
        text = text.replace(phrase, "")
    return text


def query_coverage(query: str, text: str) -> float:
    """问题关键词在资料中的覆盖率，用于 BM25-only/离线模式的拒答兜底。"""
    query_text = strip_question_phrases(query)
    query_tokens = set(zh_tokenize(query_text))
    if not query_tokens:
        return 0.0
    doc_tokens = set(zh_tokenize(text))
    return len(query_tokens & doc_tokens) / len(query_tokens)


class BM25Retriever:
    def __init__(
        self,
        chunks: list[Chunk],
        tokenize: Callable[[str], list[str]] = zh_tokenize,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.chunks = chunks
        self.tokenize = tokenize
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(chunk.text) for chunk in chunks]
        self.doc_len = [len(doc) for doc in self.docs]
        self.avgdl = sum(self.doc_len) / len(self.docs) if self.docs else 0.0
        self.term_freqs = [Counter(doc) for doc in self.docs]
        self.df: Counter[str] = Counter()
        for doc in self.docs:
            self.df.update(set(doc))
        self.n_docs = len(self.docs)

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_terms = self.tokenize(query)
        if not query_terms or not self.docs:
            return []

        scored: list[tuple[float, int]] = []
        for index, tf in enumerate(self.term_freqs):
            score = 0.0
            for term in query_terms:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                denominator = freq + self.k1 * (
                    1 - self.b
                    + self.b * self.doc_len[index] / self.avgdl
                )
                score += self._idf(term) * freq * (self.k1 + 1) / denominator
            if score > 0:
                scored.append((score, index))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            RetrievedChunk(self.chunks[index], score)
            for score, index in scored[:top_k]
        ]
