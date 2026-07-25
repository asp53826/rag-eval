"""Run a config over a fixed test set and produce a report.

The rules this enforces, which are the reason the project exists:

  - every number is tied to the exact config that produced it, by hash
  - the same config and the same data give the same numbers, always
  - reports are JSON on disk so two runs can be diffed months apart
  - per query records are kept, so "recall went down" can be followed to the
    specific queries that broke instead of being re-run blind
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field

from ragkit.corpus import Document, Query
from ragkit.pipeline import RAGPipeline, RetrievalConfig

from .metrics import (
    abstention_scores,
    exact_match,
    faithfulness,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    token_f1,
)

K_VALUES = (1, 3, 5, 10)


@dataclass
class QueryRecord:
    query_id: str
    query: str
    retrieved: list[str]
    relevant: list[str]
    answerable: bool = True
    answer: str = ""
    gold_answers: list[str] = field(default_factory=list)
    abstained: bool = False
    latency_ms: float = 0.0
    metrics: dict = field(default_factory=dict)


@dataclass
class EvalReport:
    config: dict
    config_hash: str
    dataset: str
    # quality only, and deterministic. same config plus same data must give
    # byte identical numbers here or no comparison in this repo means anything.
    metrics: dict
    # wall clock, deliberately kept out of metrics because it varies run to run
    # and would make every reproducibility check flaky
    timing: dict
    pipeline_stats: dict
    environment: dict
    records: list[dict] = field(default_factory=list)
    wall_seconds: float = 0.0
    label: str = ""

    def get(self, key: str, default=None):
        if key in self.metrics:
            return self.metrics[key]
        return self.timing.get(key, default)

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        return path

    @staticmethod
    def load(path: str) -> "EvalReport":
        with open(path) as f:
            return EvalReport(**json.load(f))


def _environment() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def evaluate_retrieval(
    cfg: RetrievalConfig,
    docs: list[Document],
    queries: list[Query],
    embedder=None,
    reranker=None,
    generator=None,
    dataset: str = "unknown",
    keep_records: bool = True,
) -> EvalReport:
    t0 = time.perf_counter()
    pipe = RAGPipeline(cfg, embedder=embedder, reranker=reranker).build(docs)

    records: list[QueryRecord] = []
    for q in queries:
        res = pipe.retrieve(q.text, q.query_id)
        # score at document level: a chunk is a means to an end, and the
        # ground truth in every real dataset is per document
        got = res.doc_ids
        rel = q.relevant_ids

        m = {}
        for k in K_VALUES:
            m[f"recall@{k}"] = recall_at_k(got, rel, k)
            m[f"precision@{k}"] = precision_at_k(got, rel, k)
            m[f"ndcg@{k}"] = ndcg_at_k(got, q.relevant, k)
            m[f"hit@{k}"] = hit_at_k(got, rel, k)
        m["mrr@10"] = mrr_at_k(got, rel, 10)

        rec = QueryRecord(
            query_id=q.query_id,
            query=q.text,
            retrieved=got[:10],
            relevant=sorted(rel),
            answerable=q.answerable,
            gold_answers=q.answers,
            latency_ms=res.latency_ms,
            metrics=m,
        )

        if generator is not None:
            out = generator.answer(q.text, res.chunks)
            rec.answer = out.text
            rec.abstained = out.abstained
            contexts = [c.full_text for c in res.chunks]
            faith, _ = faithfulness(out.text, contexts)
            rec.metrics.update(
                exact_match=exact_match(out.text, q.answers) if q.answerable else 0.0,
                token_f1=token_f1(out.text, q.answers) if q.answerable else 0.0,
                faithfulness=faith,
            )
        records.append(rec)

    agg, timing = _aggregate(records, generator is not None)
    return EvalReport(
        config=asdict(cfg),
        config_hash=cfg.fingerprint(),
        dataset=dataset,
        label=cfg.label(),
        metrics=agg,
        timing=timing,
        pipeline_stats=pipe.stats(),
        environment=_environment(),
        records=[asdict(r) for r in records] if keep_records else [],
        wall_seconds=round(time.perf_counter() - t0, 2),
    )


def _aggregate(records: list[QueryRecord], has_answers: bool) -> tuple[dict, dict]:
    if not records:
        return {}, {}
    out: dict = {}
    keys = set()
    for r in records:
        keys |= set(r.metrics)

    answerable = [r for r in records if r.answerable]
    for key in sorted(keys):
        # answer quality only means anything on questions that have an answer
        pool = answerable if key in ("exact_match", "token_f1") else records
        vals = [r.metrics[key] for r in pool if key in r.metrics]
        if vals:
            out[key] = round(sum(vals) / len(vals), 4)

    out["queries"] = len(records)
    if has_answers:
        out.update({k: round(v, 4) for k, v in abstention_scores(records).items()})

    lat = [r.latency_ms for r in records]
    timing = {
        "latency_p50_ms": round(statistics.median(lat), 2),
        "latency_p95_ms": round(percentile(lat, 95), 2),
    }
    return out, timing


def sweep(
    configs: list[RetrievalConfig],
    docs: list[Document],
    queries: list[Query],
    embedders: dict | None = None,
    rerankers: dict | None = None,
    generator=None,
    dataset: str = "unknown",
    out_dir: str | None = None,
    verbose: bool = True,
) -> list[EvalReport]:
    """Run an ablation. Embedders and rerankers are passed in already built so
    one model gets loaded once instead of once per config."""
    embedders = embedders or {}
    rerankers = rerankers or {}
    reports = []
    for i, cfg in enumerate(configs, 1):
        if verbose:
            print(f"[{i}/{len(configs)}] {cfg.label()} ... ", end="", flush=True)
        rep = evaluate_retrieval(
            cfg,
            docs,
            queries,
            embedder=embedders.get(cfg.embedder),
            reranker=rerankers.get(cfg.reranker),
            generator=generator,
            dataset=dataset,
            keep_records=bool(out_dir),
        )
        if verbose:
            print(f"nDCG@10={rep.metrics.get('ndcg@10', 0):.4f}  ({rep.wall_seconds}s)")
        if out_dir:
            rep.save(os.path.join(out_dir, f"{cfg.fingerprint()}.json"))
        reports.append(rep)
    return reports


def format_table(reports: list[EvalReport], columns: list[str] | None = None) -> str:
    columns = columns or ["ndcg@10", "recall@10", "recall@5", "mrr@10", "hit@1", "latency_p50_ms"]
    width = max(len(r.label) for r in reports) + 2
    head = "config".ljust(width) + "".join(c.rjust(14) for c in columns)
    lines = [head, "-" * len(head)]
    for r in reports:
        row = r.label.ljust(width)
        for c in columns:
            v = r.get(c)
            row += (f"{v:.4f}" if isinstance(v, float) and v < 100 else str(v)).rjust(14)
        lines.append(row)
    return "\n".join(lines)
