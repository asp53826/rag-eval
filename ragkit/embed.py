"""Embedding backends behind one interface, plus a disk cache.

The cache matters more than it sounds. An eval harness you run twenty times
while tuning chunking is useless if every run re-encodes 5000 documents, and
"I only measured it once" is how people end up shipping regressions.

Two backends:
  SentenceTransformerEmbedder   real quality, needs torch
  HashingEmbedder               no dependencies, runs in CI, deterministic

The second one exists so the whole pipeline stays testable and the harness
stays hermetic. It is genuinely worse and the eval says so, which is the point.
"""

from __future__ import annotations

import hashlib
import os
import zlib
from typing import Protocol

import numpy as np

CACHE_DIR = os.environ.get(
    "RAGKIT_CACHE",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache"),
)


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray: ...


def _fingerprint(name: str, texts: list[str]) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(name.encode())
    h.update(str(len(texts)).encode())
    for t in texts:
        h.update(t.encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()


class CachedEmbedder:
    """Wraps any embedder and memoizes whole batches on disk by content hash."""

    def __init__(self, inner: Embedder, cache_dir: str = CACHE_DIR):
        self.inner = inner
        self.cache_dir = cache_dir
        self.name = inner.name
        self.dim = inner.dim
        self.hits = 0
        self.misses = 0
        os.makedirs(cache_dir, exist_ok=True)

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        key = _fingerprint(f"{self.name}|{is_query}", texts)
        path = os.path.join(self.cache_dir, f"{key}.npy")
        if os.path.exists(path):
            self.hits += 1
            return np.load(path)
        self.misses += 1
        vecs = self.inner.encode(texts, is_query)
        # write then rename so a killed run can't leave a half written cache
        # entry that later loads as garbage. np.save wants a handle here,
        # given a path it would append .npy to the temp name.
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            np.save(f, vecs)
        os.replace(tmp, path)
        return vecs


class HashingEmbedder:
    """Hashed character n-grams projected to a dense vector.

    Roughly a random-projection sketch of a character n-gram bag. Handles
    typos and morphology better than exact term matching, but it has no idea
    that "myocardial infarction" and "heart attack" are related, and the eval
    numbers show exactly that gap.
    """

    def __init__(self, dim: int = 384, ngram: tuple[int, int] = (3, 5), seed: int = 0):
        self.name = f"hashing-{dim}"
        self.dim = dim
        self.lo, self.hi = ngram
        self.seed = seed

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        s = f" {text.lower()} "
        # crc32 rather than builtin hash(): python randomizes string hashing
        # per process, which would make embeddings differ between runs and
        # quietly poison every cached vector and every baseline
        crc = zlib.crc32
        for n in range(self.lo, self.hi + 1):
            for i in range(len(s) - n + 1):
                g = s[i : i + n]
                h = crc(g.encode("utf-8", "ignore"), self.seed) & 0x7FFFFFFF
                # signed hashing keeps collisions from all pushing one way
                v[h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 batch_size: int = 64, query_prefix: str = "", doc_prefix: str = ""):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.name = model_name.split("/")[-1]
        self.dim = self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size
        # e5/bge style models want asymmetric prefixes, minilm doesn't
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        prefix = self.query_prefix if is_query else self.doc_prefix
        if prefix:
            texts = [prefix + t for t in texts]
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)


def get_embedder(spec: str = "minilm", cached: bool = True) -> Embedder:
    if spec in ("hashing", "hash"):
        inner: Embedder = HashingEmbedder()
    elif spec == "minilm":
        inner = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    else:
        inner = SentenceTransformerEmbedder(spec)
    return CachedEmbedder(inner) if cached else inner
