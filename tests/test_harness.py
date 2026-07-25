"""Tests for the harness, which is the part of this repo that has to be right.

A benchmark you can't reproduce is an anecdote, and a regression check that
doesn't fail on a regression is worse than none at all because it buys false
confidence.
"""

import json

import pytest

from eval.harness import evaluate_retrieval, format_table
from eval.regression import check_floors, compare
from ragkit.corpus import Document, Query
from ragkit.embed import HashingEmbedder
from ragkit.generate import AlwaysAnswerGenerator, ExtractiveGenerator
from ragkit.pipeline import RetrievalConfig

DOCS = [
    Document("d1", "The mitochondrion is the powerhouse of the cell and produces ATP.", "Biology"),
    Document("d2", "Kubernetes schedules containers across a cluster of worker nodes.", "K8s"),
    Document("d3", "Photosynthesis converts light into chemical energy inside chloroplasts.", "Plants"),
    Document("d4", "A pod is the smallest deployable unit in Kubernetes.", "Pods"),
    Document("d5", "Rust uses ownership and borrowing to guarantee memory safety.", "Rust"),
]

QUERIES = [
    Query("q1", "what produces ATP in the cell", {"d1": 1}, ["mitochondrion"]),
    Query("q2", "smallest deployable unit in kubernetes", {"d4": 1}, ["a pod"]),
    Query("q3", "how does rust guarantee memory safety", {"d5": 1}, ["ownership and borrowing"]),
    # shares vocabulary with the corpus but has no answer in it, which is how
    # real unanswerable questions look. a query with no overlapping terms at
    # all retrieves nothing and abstaining is then trivially correct.
    Query("q4", "which kubernetes release first shipped pod autoscaling",
          {}, [], answerable=False),
]


def run(cfg=None, generator=None, queries=None):
    cfg = cfg or RetrievalConfig(retriever="bm25", top_k=3)
    return evaluate_retrieval(
        cfg, DOCS, queries or QUERIES,
        embedder=HashingEmbedder(), generator=generator, dataset="unit",
    )


def test_same_config_gives_identical_numbers():
    """Reproducibility is the whole premise. If two runs of the same config
    disagree, no comparison between two different configs means anything."""
    a, b = run(), run()
    assert a.metrics == b.metrics
    assert a.config_hash == b.config_hash


def test_dense_pipeline_is_reproducible_too():
    cfg = RetrievalConfig(retriever="hybrid", embedder="hashing", fusion="weighted", top_k=3)
    assert run(cfg).metrics == run(cfg).metrics


def test_report_records_the_config_that_produced_it():
    rep = run()
    assert rep.config["retriever"] == "bm25"
    assert rep.config_hash and rep.dataset == "unit"
    assert rep.environment["python"]


def test_report_roundtrips_through_json(tmp_path):
    from eval.harness import EvalReport

    rep = run()
    path = rep.save(str(tmp_path / "r.json"))
    again = EvalReport.load(path)
    assert again.metrics == rep.metrics
    assert again.config_hash == rep.config_hash
    assert RetrievalConfig(**again.config).fingerprint() == rep.config_hash


def test_retrieval_actually_works_on_the_toy_set():
    rep = run()
    assert rep.metrics["hit@1"] >= 0.75  # 3 of 4, q4 has no relevant doc
    assert rep.metrics["recall@5"] >= 0.75


def test_per_query_records_are_kept():
    rep = run()
    assert len(rep.records) == len(QUERIES)
    assert {r["query_id"] for r in rep.records} == {"q1", "q2", "q3", "q4"}
    assert all("metrics" in r for r in rep.records)


def test_answer_metrics_only_count_answerable_questions():
    """An unanswerable question has no gold answer, so scoring it as EM=0
    would silently drag the answer metrics down for doing the right thing."""
    rep = run(generator=ExtractiveGenerator(threshold=-1.0))
    answerable = [r for r in rep.records if r["answerable"]]
    manual = sum(r["metrics"]["token_f1"] for r in answerable) / len(answerable)
    assert rep.metrics["token_f1"] == pytest.approx(round(manual, 4), abs=1e-3)


def test_always_answering_maximizes_hallucination():
    rep = run(generator=AlwaysAnswerGenerator())
    assert rep.metrics["hallucination_rate"] == 1.0
    assert rep.metrics["over_abstention_rate"] == 0.0


def test_abstaining_always_removes_hallucination_entirely():
    rep = run(generator=ExtractiveGenerator(threshold=99.0))
    assert rep.metrics["hallucination_rate"] == 0.0
    assert rep.metrics["over_abstention_rate"] == 1.0


def test_faithfulness_is_reported_when_generating():
    rep = run(generator=ExtractiveGenerator(threshold=-1.0))
    assert 0.0 <= rep.metrics["faithfulness"] <= 1.0


# ---------- regression detection ----------

def test_compare_flags_a_real_drop():
    base = run()
    worse = run()
    worse.metrics["ndcg@10"] = base.metrics["ndcg@10"] - 0.05
    cmp = compare(base, worse)
    assert not cmp.ok
    assert any(d.metric == "ndcg@10" and d.regressed for d in cmp.diffs)


def test_compare_tolerates_noise_below_the_guard():
    base = run()
    jittered = run()
    jittered.metrics["ndcg@10"] = base.metrics["ndcg@10"] - 0.001
    assert compare(base, jittered).ok


def test_compare_does_not_flag_improvements():
    base = run()
    better = run()
    better.metrics["ndcg@10"] = min(1.0, base.metrics["ndcg@10"] + 0.2)
    cmp = compare(base, better)
    assert cmp.ok
    assert any(d.delta > 0 for d in cmp.diffs)


def test_lower_is_better_metrics_regress_upward():
    """Latency and hallucination going up is bad news, not good news."""
    base = run(generator=ExtractiveGenerator())
    worse = run(generator=ExtractiveGenerator())
    base.metrics["hallucination_rate"] = 0.1
    worse.metrics["hallucination_rate"] = 0.9
    cmp = compare(base, worse, guards={"hallucination_rate": 0.05})
    assert not cmp.ok


def test_compare_notices_the_config_changed():
    base = run(RetrievalConfig(retriever="bm25", top_k=3))
    other = run(RetrievalConfig(retriever="bm25", top_k=5))
    assert compare(base, other).config_changed


def test_compare_names_the_queries_that_broke():
    base = run()
    broken = run()
    for r in broken.records:
        if r["query_id"] == "q1":
            r["metrics"]["hit@1"] = 0.0
    cmp = compare(base, broken)
    assert "q1" in cmp.broken_queries


def test_floors_catch_a_regenerated_baseline():
    """Absolute floors exist because the easiest way to make a failing
    regression check pass is to re-freeze the baseline."""
    rep = run()
    assert check_floors(rep, {"ndcg@10": 0.0}) == []
    assert check_floors(rep, {"ndcg@10": 0.99})
    assert check_floors(rep, {"nonexistent_metric": 0.5})


def test_floors_treat_ceilings_correctly():
    rep = run(generator=AlwaysAnswerGenerator())
    assert check_floors(rep, {"hallucination_rate": 0.5})  # 1.0 is over the ceiling
    assert check_floors(rep, {"hallucination_rate": 1.0}) == []


def test_format_table_renders_every_config():
    reps = [run(RetrievalConfig(retriever="bm25")), run(RetrievalConfig(retriever="dense", embedder="hashing"))]
    out = format_table(reps)
    assert "bm25" in out and "dense" in out and "ndcg@10" in out
