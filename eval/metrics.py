"""Metrics.

Retrieval metrics are the standard TREC definitions so the numbers can be
compared against published results. Answer metrics are SQuAD's official
normalization, again so they mean the same thing they mean elsewhere.

Everything here is deterministic. An LLM judge is available separately for
faithfulness, but the metrics that gate CI don't depend on one, because a
regression suite whose baseline drifts with a vendor's model isn't a
regression suite.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter


# ---------- retrieval ----------

def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / k


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def mrr_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    for i, doc in enumerate(retrieved[:k], start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], graded: dict[str, int], k: int) -> float:
    """Graded nDCG with the standard 2^rel - 1 gain, as TREC and BEIR use."""
    dcg = 0.0
    for i, doc in enumerate(retrieved[:k]):
        rel = graded.get(doc, 0)
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(i + 2)
    ideal = sorted(graded.values(), reverse=True)[:k]
    idcg = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal) if rel > 0)
    return dcg / idcg if idcg else 0.0


# ---------- answers ----------

def normalize_answer(s: str) -> str:
    """SQuAD's official normalization: lowercase, strip articles and punctuation."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, golds: list[str]) -> float:
    if not golds:
        return 0.0
    p = normalize_answer(pred)
    return 1.0 if any(p == normalize_answer(g) for g in golds) else 0.0


def token_f1(pred: str, golds: list[str]) -> float:
    if not golds:
        return 0.0
    return max(_f1(pred, g) for g in golds)


def _f1(pred: str, gold: str) -> float:
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(p)
    recall = same / len(g)
    return 2 * precision * recall / (precision + recall)


# ---------- grounding ----------

def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def claim_support(claim: str, context: str, n: int = 3, threshold: float = 0.6) -> float:
    """How much of a claim's wording is actually present in the context.

    This is a grounding proxy, not entailment. It catches the failure that
    matters most in practice, a model asserting specifics the retrieved
    passages never mentioned, and it does it without an API call. An LLM judge
    is available in judge.py for the cases where phrasing differs from
    substance.
    """
    c = normalize_answer(claim).split()
    ctx = normalize_answer(context).split()
    if not c:
        return 1.0
    if len(c) < n:
        return 1.0 if " ".join(c) in " ".join(ctx) else 0.0
    cg = _ngrams(c, n)
    ctxg = _ngrams(ctx, n)
    return len(cg & ctxg) / len(cg)


def faithfulness(answer: str, contexts: list[str], n: int = 3,
                 threshold: float = 0.6) -> tuple[float, list[str]]:
    """Returns (share of supported sentences, the unsupported ones)."""
    from ragkit.text import split_sentences

    context = "\n".join(contexts)
    sents = [s for s in split_sentences(answer) if len(normalize_answer(s).split()) >= 3]
    if not sents:
        return 1.0, []
    unsupported = [s for s in sents if claim_support(s, context, n) < threshold]
    return 1 - len(unsupported) / len(sents), unsupported


# ---------- abstention ----------

def abstention_scores(records) -> dict:
    """The four-way breakdown on answerable vs unanswerable questions.

    A system that answers everything scores perfectly on answerable questions
    and hallucinates on every unanswerable one. Reporting a single accuracy
    number hides that completely, so it gets split out.
    """
    tp = fp = tn = fn = 0
    for r in records:
        abstained = r.abstained
        if r.answerable:
            if abstained:
                fn += 1  # bailed on a question it could have answered
            else:
                tp += 1
        else:
            if abstained:
                tn += 1  # correctly declined
            else:
                fp += 1  # answered something the corpus doesn't support
    total = tp + fp + tn + fn
    answerable = tp + fn
    unanswerable = tn + fp
    return {
        "abstention_precision": tn / (tn + fn) if (tn + fn) else 0.0,
        "abstention_recall": tn / unanswerable if unanswerable else 0.0,
        "hallucination_rate": fp / unanswerable if unanswerable else 0.0,
        "over_abstention_rate": fn / answerable if answerable else 0.0,
        "answer_accuracy": (tp + tn) / total if total else 0.0,
    }


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = (len(s) - 1) * p / 100
    lo, hi = int(i), min(int(i) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)
