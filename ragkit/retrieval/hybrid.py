"""Combining a lexical and a dense ranking.

Two ways, because they fail differently and the harness should be able to say
which one wins on a given corpus rather than the README asserting it:

  RRF          rank based, scale free, one parameter, hard to get wrong
  weighted     score based, needs normalization, can beat RRF when tuned and
               can quietly fall apart when the two score distributions drift

Lexical retrieval nails exact terms, dosages, identifiers, rare proper nouns.
Dense retrieval handles the case where the user and the document simply don't
use the same words. Most real queries need a bit of both.
"""

from __future__ import annotations

import numpy as np


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60,
                           weights: list[float] | None = None) -> list[tuple[str, float]]:
    """score(d) = sum_i w_i / (k + rank_i(d)).

    k=60 is the value from the original RRF paper. It flattens the top of the
    curve so rank 1 and rank 2 aren't wildly far apart, which stops one
    retriever's confident mistake from dominating the fusion.
    """
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def weighted_fusion(score_lists: list[np.ndarray], ids: list[str],
                    weights: list[float], k: int = 10) -> list[tuple[str, float]]:
    """Min-max normalize each score vector, then take a weighted sum.

    Normalization is per query on purpose. BM25 scores have no fixed range and
    cosine sits in [-1, 1], so combining raw scores just measures whichever
    retriever happens to produce bigger numbers.
    """
    total = np.zeros(len(ids), dtype=np.float32)
    for scores, w in zip(score_lists, weights):
        total += w * _minmax(scores.astype(np.float32))
    k = min(k, total.size)
    if k == 0:
        return []
    top = np.argpartition(-total, k - 1)[:k]
    top = top[np.argsort(-total[top])]
    return [(ids[i], float(total[i])) for i in top]


class HybridRetriever:
    def __init__(self, bm25, dense, mode: str = "rrf", rrf_k: int = 60,
                 weights: tuple[float, float] = (0.5, 0.5), candidates: int = 100):
        self.bm25 = bm25
        self.dense = dense
        self.mode = mode
        self.rrf_k = rrf_k
        self.weights = weights
        # how deep each retriever goes before fusion. too shallow and RRF
        # never sees the document one side found and the other missed.
        self.candidates = candidates

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        if self.mode == "rrf":
            lex = [c for c, _ in self.bm25.search(query, self.candidates)]
            den = [c for c, _ in self.dense.search(query, self.candidates)]
            return reciprocal_rank_fusion([lex, den], self.rrf_k, list(self.weights))[:k]
        if self.mode == "weighted":
            lex = self.bm25.score_all(query)
            den = self.dense.score_all(query)
            return weighted_fusion([lex, den], self.bm25.chunk_ids, list(self.weights), k)
        raise ValueError(f"unknown fusion mode {self.mode}")
