"""Wiring the pieces into one configurable pipeline.

Everything that could change a number is on RetrievalConfig, and the config
hashes into the eval report. That's the whole discipline: if you can't tell
which knobs produced a result, the result isn't a measurement.

  rewrite -> retrieve (lexical | dense | hybrid) -> fuse variants -> rerank
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field

from .chunking import chunk_corpus, get_chunker
from .corpus import Chunk, Document
from .rerank import NoopReranker
from .retrieval.bm25 import BM25
from .retrieval.dense import DenseIndex
from .retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from .rewrite import get_rewriter


@dataclass
class RetrievalConfig:
    chunker: str = "whole"
    retriever: str = "hybrid"  # bm25 | dense | hybrid
    embedder: str = "minilm"
    fusion: str = "rrf"  # rrf | weighted
    rrf_k: int = 60
    weights: tuple[float, float] = (0.5, 0.5)
    candidates: int = 100
    rewriter: str = "none"
    reranker: str = "none"  # none | cross-encoder | lexical
    rerank_candidates: int = 50
    top_k: int = 10
    bm25_k1: float = 0.9
    bm25_b: float = 0.4
    stemming: bool = True

    def __post_init__(self):
        # json turns tuples into lists, so a config loaded back from a report
        # would otherwise not equal the one that produced it
        self.weights = tuple(self.weights)

    def fingerprint(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.blake2b(blob.encode(), digest_size=8).hexdigest()

    def label(self) -> str:
        bits = [self.retriever]
        if self.retriever in ("dense", "hybrid"):
            bits.append(self.embedder)
        if self.retriever == "hybrid":
            bits.append(self.fusion)
            if self.fusion == "weighted":
                bits.append(f"w={self.weights[0]:g}/{self.weights[1]:g}")
        if self.chunker != "whole":
            bits.append(self.chunker)
        if self.rewriter != "none":
            bits.append(f"rw:{self.rewriter}")
        if self.reranker != "none":
            bits.append(f"rr:{self.reranker}")
        return " + ".join(bits)


@dataclass
class RetrievalResult:
    query_id: str
    query: str
    chunk_ids: list[str]
    doc_ids: list[str]
    scores: list[float]
    chunks: list[Chunk] = field(default_factory=list)
    latency_ms: float = 0.0


class RAGPipeline:
    def __init__(self, cfg: RetrievalConfig, embedder=None, reranker=None):
        self.cfg = cfg
        self.chunks: list[Chunk] = []
        self.by_id: dict[str, Chunk] = {}
        self.bm25: BM25 | None = None
        self.dense: DenseIndex | None = None
        self.hybrid: HybridRetriever | None = None
        self.rewriter = get_rewriter(cfg.rewriter)
        self.reranker = reranker or NoopReranker()
        self._embedder = embedder
        self.index_seconds = 0.0

    def build(self, docs: list[Document]) -> "RAGPipeline":
        t0 = time.perf_counter()
        self.chunks = chunk_corpus(docs, get_chunker(self.cfg.chunker))
        self.by_id = {c.chunk_id: c for c in self.chunks}

        if self.cfg.retriever in ("bm25", "hybrid"):
            self.bm25 = BM25(self.cfg.bm25_k1, self.cfg.bm25_b, self.cfg.stemming).index(self.chunks)
        if self.cfg.retriever in ("dense", "hybrid"):
            if self._embedder is None:
                from .embed import get_embedder

                self._embedder = get_embedder(self.cfg.embedder)
            self.dense = DenseIndex(self._embedder).index(self.chunks)
        if self.cfg.retriever == "hybrid":
            # weighted fusion adds score vectors positionally, so both indexes
            # have to be built from the same chunk list in the same order
            assert self.bm25.chunk_ids == self.dense.chunk_ids
            self.hybrid = HybridRetriever(
                self.bm25, self.dense, self.cfg.fusion, self.cfg.rrf_k,
                self.cfg.weights, self.cfg.candidates,
            )
        self.index_seconds = time.perf_counter() - t0
        return self

    def _retrieve_one(self, query: str, k: int) -> list[tuple[str, float]]:
        if self.cfg.retriever == "bm25":
            return self.bm25.search(query, k)
        if self.cfg.retriever == "dense":
            return self.dense.search(query, k)
        return self.hybrid.search(query, k)

    def retrieve(self, query: str, query_id: str = "") -> RetrievalResult:
        t0 = time.perf_counter()
        variants = self.rewriter.rewrite(query)
        depth = max(self.cfg.rerank_candidates, self.cfg.top_k) if self._reranking else self.cfg.top_k

        if len(variants) == 1:
            hits = self._retrieve_one(variants[0], depth)
        else:
            # fuse the variants' rankings so a bad rewrite can't hijack the list
            rankings = [[c for c, _ in self._retrieve_one(v, depth)] for v in variants]
            weights = [1.0] + [0.5] * (len(rankings) - 1)
            hits = reciprocal_rank_fusion(rankings, self.cfg.rrf_k, weights)[:depth]

        if self._reranking:
            cands = [self.by_id[c] for c, _ in hits]
            hits = self.reranker.rerank(query, cands, self.cfg.top_k)

        hits = hits[: self.cfg.top_k]
        chunks = [self.by_id[c] for c, _ in hits]
        return RetrievalResult(
            query_id=query_id,
            query=query,
            chunk_ids=[c for c, _ in hits],
            doc_ids=_dedupe([c.doc_id for c in chunks]),
            scores=[s for _, s in hits],
            chunks=chunks,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    @property
    def _reranking(self) -> bool:
        return self.reranker.name != "none"

    def stats(self) -> dict:
        return {
            "chunks": len(self.chunks),
            "chunks_per_doc": round(len(self.chunks) / max(len({c.doc_id for c in self.chunks}), 1), 2),
            "index_seconds": round(self.index_seconds, 2),
            "bm25_vocab": self.bm25.vocab_size if self.bm25 else 0,
        }


def _dedupe(xs: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
