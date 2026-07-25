"""Comparing an eval run against a stored baseline.

This is what makes the difference between "we measured it once" and a test
suite. A retrieval change that costs 3 points of recall doesn't crash, doesn't
throw, and doesn't show up in any unit test. The only way it gets caught is if
something compares today's number to the number from before the change and
fails the build.

Guards are per metric, because "recall dropped 2 points" and "p95 latency rose
2 ms" are not the same news.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from .harness import EvalReport

# metric -> how far it may fall before the build fails
DEFAULT_GUARDS = {
    "ndcg@10": 0.01,
    "recall@10": 0.01,
    "recall@5": 0.01,
    "mrr@10": 0.01,
    "hit@1": 0.02,
    "token_f1": 0.02,
    "faithfulness": 0.02,
}

# metrics where smaller is better
LOWER_IS_BETTER = {"hallucination_rate", "over_abstention_rate", "latency_p50_ms", "latency_p95_ms"}


@dataclass
class MetricDiff:
    metric: str
    baseline: float
    current: float
    delta: float
    tolerance: float
    regressed: bool

    @property
    def pct(self) -> float:
        return (self.delta / self.baseline * 100) if self.baseline else 0.0


@dataclass
class Comparison:
    label: str
    diffs: list[MetricDiff] = field(default_factory=list)
    new_queries: list[str] = field(default_factory=list)
    fixed_queries: list[str] = field(default_factory=list)
    broken_queries: list[str] = field(default_factory=list)
    config_changed: bool = False

    @property
    def regressions(self) -> list[MetricDiff]:
        return [d for d in self.diffs if d.regressed]

    @property
    def ok(self) -> bool:
        return not self.regressions


def compare(baseline: EvalReport, current: EvalReport,
            guards: dict[str, float] | None = None) -> Comparison:
    guards = {**DEFAULT_GUARDS, **(guards or {})}
    cmp = Comparison(label=current.label or current.config_hash)
    cmp.config_changed = baseline.config_hash != current.config_hash

    for metric, tol in sorted(guards.items()):
        b, c = baseline.get(metric), current.get(metric)
        if b is None or c is None:
            continue
        delta = c - b
        regressed = (-delta > tol) if metric not in LOWER_IS_BETTER else (delta > tol)
        cmp.diffs.append(MetricDiff(metric, b, c, round(delta, 4), tol, regressed))

    cmp.broken_queries, cmp.fixed_queries = _query_moves(baseline, current)
    return cmp


def _query_moves(baseline: EvalReport, current: EvalReport, metric: str = "hit@1"):
    """Which individual queries changed verdict.

    An aggregate that holds steady can still hide ten queries breaking and ten
    others starting to work. This is the first thing you want when a number
    moves and you have no idea why.
    """
    b = {r["query_id"]: r["metrics"].get(metric, 0.0) for r in baseline.records}
    c = {r["query_id"]: r["metrics"].get(metric, 0.0) for r in current.records}
    broken = sorted(q for q in b.keys() & c.keys() if b[q] > 0 and c[q] == 0)
    fixed = sorted(q for q in b.keys() & c.keys() if b[q] == 0 and c[q] > 0)
    return broken, fixed


def format_comparison(cmp: Comparison, show_queries: int = 5) -> str:
    lines = [f"baseline vs current: {cmp.label}"]
    if cmp.config_changed:
        lines.append("  note: config hash differs, this is not a like for like rerun")
    lines.append(f"  {'metric':<22}{'baseline':>10}{'current':>10}{'delta':>10}  status")
    lines.append("  " + "-" * 60)
    for d in cmp.diffs:
        status = "REGRESSED" if d.regressed else ("improved" if d.delta > 0 else "ok")
        lines.append(
            f"  {d.metric:<22}{d.baseline:>10.4f}{d.current:>10.4f}{d.delta:>+10.4f}  {status}"
        )
    if cmp.broken_queries:
        shown = ", ".join(cmp.broken_queries[:show_queries])
        lines.append(f"  newly failing ({len(cmp.broken_queries)}): {shown}")
    if cmp.fixed_queries:
        shown = ", ".join(cmp.fixed_queries[:show_queries])
        lines.append(f"  newly passing ({len(cmp.fixed_queries)}): {shown}")
    lines.append(f"  => {'PASS' if cmp.ok else 'FAIL'}")
    return "\n".join(lines)


def save_baseline(report: EvalReport, path: str) -> str:
    return report.save(path)


def load_baseline(path: str) -> EvalReport | None:
    if not os.path.exists(path):
        return None
    return EvalReport.load(path)


def check_floors(report: EvalReport, floors: dict[str, float]) -> list[str]:
    """Absolute minimums, independent of any baseline.

    Belt and braces for the case where somebody regenerates the baseline to
    make a failing build go green.
    """
    failures = []
    for metric, floor in sorted(floors.items()):
        v = report.get(metric)
        if v is None:
            failures.append(f"{metric} missing from report")
        elif metric in LOWER_IS_BETTER:
            if v > floor:
                failures.append(f"{metric}={v:.4f} above ceiling {floor}")
        elif v < floor:
            failures.append(f"{metric}={v:.4f} below floor {floor}")
    return failures


def load_floors(path: str) -> dict[str, float]:
    with open(path) as f:
        return json.load(f)
