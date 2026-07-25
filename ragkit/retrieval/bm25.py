"""BM25 Okapi.

Written out rather than pulled in, partly because it's short and partly
because the parameters (k1, b, stemming, stopwords) are exactly the knobs the
eval harness exists to measure. Hard to ablate what you can't reach.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from ..corpus import Chunk
from ..text import tokenize


class BM25:
    def __init__(self, k1: float = 0.9, b: float = 0.4, stemming: bool = True,
                 remove_stopwords: bool = True):
        # k1=0.9 b=0.4 are the BEIR/Anserini defaults, not the k1=1.2 b=0.75
        # textbook ones. worth several points on short-document collections.
        self.k1 = k1
        self.b = b
        self.stemming = stemming
        self.remove_stopwords = remove_stopwords

        self.chunk_ids: list[str] = []
        self.doc_len = np.zeros(0, dtype=np.float32)
        self.avgdl = 0.0
        # term -> (doc indices, term frequencies)
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.idf: dict[str, float] = {}

    def _tok(self, text: str) -> list[str]:
        return tokenize(text, self.remove_stopwords, self.stemming)

    def index(self, chunks: list[Chunk]) -> "BM25":
        self.chunk_ids = [c.chunk_id for c in chunks]
        n = len(chunks)
        lens = np.zeros(n, dtype=np.float32)
        raw: dict[str, list[tuple[int, int]]] = {}

        for i, c in enumerate(chunks):
            toks = self._tok(c.full_text)
            lens[i] = len(toks)
            for term, tf in Counter(toks).items():
                raw.setdefault(term, []).append((i, tf))

        self.doc_len = lens
        self.avgdl = float(lens.mean()) if n else 0.0
        for term, plist in raw.items():
            idx = np.fromiter((p[0] for p in plist), dtype=np.int32, count=len(plist))
            tfs = np.fromiter((p[1] for p in plist), dtype=np.float32, count=len(plist))
            self.postings[term] = (idx, tfs)
            df = len(plist)
            self.idf[term] = math.log(1 + (n - df + 0.5) / (df + 0.5))
        return self

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        scores = self.score_all(query)
        if not scores.size:
            return []
        k = min(k, scores.size)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.chunk_ids[i], float(scores[i])) for i in top if scores[i] > 0]

    def score_all(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.chunk_ids), dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / (self.avgdl or 1.0))
        for term in self._tok(query):
            post = self.postings.get(term)
            if post is None:
                continue
            idx, tf = post
            scores[idx] += self.idf[term] * (tf * (self.k1 + 1)) / (tf + norm[idx])
        return scores

    @property
    def vocab_size(self) -> int:
        return len(self.postings)
