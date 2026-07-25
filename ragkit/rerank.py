"""Reranking the candidate list.

First stage retrieval has to score every chunk, so it can only afford to look
at query and document separately. A cross encoder reads them together, which
lets it catch things a dot product structurally cannot ("does this passage
actually answer the question, or does it merely share vocabulary with it").
It's far too slow to run over a whole corpus, which is exactly why it runs on
the top 50 instead.

MMR is a different tool for a different problem: not relevance but redundancy.
Five near-identical chunks in a context window is four wasted slots.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .corpus import Chunk


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: list[Chunk], k: int) -> list[tuple[str, float]]: ...


class NoopReranker:
    name = "none"

    def rerank(self, query: str, candidates: list[Chunk], k: int) -> list[tuple[str, float]]:
        return [(c.chunk_id, 1.0 / (i + 1)) for i, c in enumerate(candidates[:k])]


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 batch_size: int = 64, max_candidates: int = 50):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, max_length=512)
        self.name = model_name.split("/")[-1]
        self.batch_size = batch_size
        self.max_candidates = max_candidates

    def rerank(self, query: str, candidates: list[Chunk], k: int) -> list[tuple[str, float]]:
        cands = candidates[: self.max_candidates]
        if not cands:
            return []
        pairs = [(query, c.full_text) for c in cands]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        order = np.argsort(-np.asarray(scores))[:k]
        return [(cands[i].chunk_id, float(scores[i])) for i in order]


class LexicalOverlapReranker:
    """Dependency-free stand-in so reranking stays in the CI ablation.

    Scores query term coverage weighted toward rarer terms. Nowhere near a
    cross encoder, and the eval table is where you go to see by how much.
    """

    name = "lexical-overlap"

    def __init__(self, max_candidates: int = 50):
        self.max_candidates = max_candidates

    def rerank(self, query: str, candidates: list[Chunk], k: int) -> list[tuple[str, float]]:
        from .text import tokenize

        cands = candidates[: self.max_candidates]
        if not cands:
            return []
        q = set(tokenize(query))
        df: dict[str, int] = {}
        toks = []
        for c in cands:
            t = set(tokenize(c.full_text))
            toks.append(t)
            for term in t & q:
                df[term] = df.get(term, 0) + 1

        n = len(cands)
        scored = []
        for c, t in zip(cands, toks):
            hit = t & q
            score = sum(np.log(1 + n / df[term]) for term in hit) / (len(q) or 1)
            scored.append((c.chunk_id, float(score)))
        scored.sort(key=lambda kv: -kv[1])
        return scored[:k]


def mmr(query_vec: np.ndarray, cand_vecs: np.ndarray, cand_ids: list[str],
        k: int, lam: float = 0.7) -> list[tuple[str, float]]:
    """Maximal marginal relevance.

    Greedily pick the candidate maximizing lam * relevance minus
    (1 - lam) * similarity to whatever is already selected. lam=1 is plain
    relevance ranking, lam=0 picks for diversity alone.
    """
    if not cand_ids:
        return []
    rel = cand_vecs @ query_vec
    picked: list[int] = []
    picked_scores: list[float] = []
    remaining = list(range(len(cand_ids)))

    while remaining and len(picked) < k:
        if not picked:
            best = max(remaining, key=lambda i: rel[i])
            score = float(rel[best])
        else:
            sim = cand_vecs[remaining] @ cand_vecs[picked].T
            worst = sim.max(axis=1)
            vals = lam * rel[remaining] - (1 - lam) * worst
            pos = int(np.argmax(vals))
            best = remaining[pos]
            score = float(vals[pos])
        picked.append(best)
        picked_scores.append(score)
        remaining.remove(best)

    return [(cand_ids[i], s) for i, s in zip(picked, picked_scores)]
