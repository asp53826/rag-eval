"""Calibrating the abstention threshold, and checking whether it's calibratable.

Picking a threshold by trying values until the test number looks good is how
you end up shipping a system tuned to its own benchmark. So: sweep on a dev
split, choose there, report on test, never the other way round.

The prior question is whether the confidence signal separates the two classes
at all. If its AUC is 0.5 the signal carries no information and *no* threshold
will work, which is worth knowing before spending a week tuning one.
"""

from __future__ import annotations

from dataclasses import dataclass


def roc_auc(scores: list[float], positive: list[bool]) -> float:
    """AUC via the rank sum identity, with ties averaged.

    0.5 means the score is useless for telling the classes apart.
    """
    n_pos = sum(positive)
    n_neg = len(positive) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1

    pos_rank_sum = sum(r for r, p in zip(ranks, positive) if p)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


@dataclass
class ThresholdPoint:
    threshold: float
    hallucination_rate: float  # answered when it should have abstained
    over_abstention_rate: float  # abstained when it could have answered
    answer_accuracy: float
    abstain_share: float


def abstention_curve(confidences: list[float], answerable: list[bool],
                     steps: int = 41) -> list[ThresholdPoint]:
    """Sweep every threshold and report both error types at each one.

    Two error types that trade off against each other cannot be summarized by
    one number. A curve makes the choice explicit instead of hiding it in a
    default.
    """
    if not confidences:
        return []
    lo, hi = min(confidences), max(confidences)
    if hi == lo:
        hi = lo + 1e-6
    out = []
    for s in range(steps):
        t = lo + (hi - lo) * s / (steps - 1)
        fp = fn = tp = tn = 0
        for c, ans in zip(confidences, answerable):
            abstained = c < t
            if ans:
                fn += abstained
                tp += not abstained
            else:
                tn += abstained
                fp += not abstained
        n_ans = tp + fn
        n_un = tn + fp
        out.append(
            ThresholdPoint(
                threshold=round(t, 4),
                hallucination_rate=round(fp / n_un, 4) if n_un else 0.0,
                over_abstention_rate=round(fn / n_ans, 4) if n_ans else 0.0,
                answer_accuracy=round((tp + tn) / len(confidences), 4),
                abstain_share=round((tn + fn) / len(confidences), 4),
            )
        )
    return out


def pick_threshold(curve: list[ThresholdPoint], objective: str = "accuracy",
                   max_hallucination: float = 0.2) -> float:
    """objective 'accuracy' maximizes overall correctness.

    'safe' instead maximizes how many questions get answered subject to a hard
    ceiling on the hallucination rate, which is the shape most real products
    actually want.
    """
    if not curve:
        return 0.0
    if objective == "accuracy":
        return max(curve, key=lambda p: p.answer_accuracy).threshold
    if objective == "safe":
        ok = [p for p in curve if p.hallucination_rate <= max_hallucination]
        if not ok:
            return max(curve, key=lambda p: -p.hallucination_rate).threshold
        return min(ok, key=lambda p: p.over_abstention_rate).threshold
    raise ValueError(f"unknown objective {objective}")


def split_dev_test(queries, dev_fraction: float = 0.4, seed: int = 0):
    """Stratified by answerability, so the dev split can't end up with a
    different answerable/unanswerable balance than test."""
    import random

    rng = random.Random(seed)
    ans = [q for q in queries if q.answerable]
    un = [q for q in queries if not q.answerable]
    rng.shuffle(ans)
    rng.shuffle(un)
    a = int(len(ans) * dev_fraction)
    u = int(len(un) * dev_fraction)
    dev = ans[:a] + un[:u]
    test = ans[a:] + un[u:]
    rng.shuffle(dev)
    rng.shuffle(test)
    return dev, test


def summarize_signal(confidences: list[float], answerable: list[bool]) -> dict:
    auc = roc_auc(confidences, answerable)
    ans = [c for c, a in zip(confidences, answerable) if a]
    un = [c for c, a in zip(confidences, answerable) if not a]
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return {
        "abstention_signal_auc": round(auc, 4),
        "mean_conf_answerable": round(mean(ans), 4),
        "mean_conf_unanswerable": round(mean(un), 4),
        "separation": round(mean(ans) - mean(un), 4),
        # anything under about 0.6 means the threshold is not worth tuning,
        # the signal itself has to change
        "signal_usable": bool(auc >= 0.6),
    }
