"""Exact cosine search over normalized embeddings.

Deliberately brute force. At 5k-50k chunks a full matmul is milliseconds and
an approximate index would only add recall loss for no benefit, which would
muddy every retrieval number this repo reports. Approximate search is its own
problem with its own tradeoff curve.
"""

from __future__ import annotations

import numpy as np

from ..corpus import Chunk
from ..embed import Embedder


class DenseIndex:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.chunk_ids: list[str] = []
        self.vectors = np.zeros((0, embedder.dim), dtype=np.float32)

    def index(self, chunks: list[Chunk]) -> "DenseIndex":
        self.chunk_ids = [c.chunk_id for c in chunks]
        self.vectors = self.embedder.encode([c.full_text for c in chunks], is_query=False)
        # cosine becomes a dot product once everything is unit length
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors = self.vectors / np.maximum(norms, 1e-9)
        return self

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        scores = self.score_all(query)
        if not scores.size:
            return []
        k = min(k, scores.size)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.chunk_ids[i], float(scores[i])) for i in top]

    def score_all(self, query: str) -> np.ndarray:
        if not self.chunk_ids:
            return np.zeros(0, dtype=np.float32)
        q = self.embedder.encode([query], is_query=True)[0]
        q = q / max(float(np.linalg.norm(q)), 1e-9)
        return self.vectors @ q

    def search_batch(self, queries: list[str], k: int = 10) -> list[list[tuple[str, float]]]:
        """One matmul for the whole query set, which is what makes sweeping
        over 300 queries x 12 configs finish in seconds instead of minutes."""
        if not self.chunk_ids or not queries:
            return [[] for _ in queries]
        q = self.embedder.encode(queries, is_query=True)
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
        scores = q @ self.vectors.T
        k = min(k, scores.shape[1])
        idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        out = []
        for r in range(scores.shape[0]):
            row = idx[r][np.argsort(-scores[r, idx[r]])]
            out.append([(self.chunk_ids[i], float(scores[r, i])) for i in row])
        return out
