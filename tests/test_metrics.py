"""Metrics have to be right before anything they measure can be believed.

Checked against hand-computed values, not against themselves.
"""

import math

import pytest

from eval.calibrate import abstention_curve, pick_threshold, roc_auc
from eval.metrics import (
    exact_match,
    faithfulness,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    normalize_answer,
    precision_at_k,
    recall_at_k,
    token_f1,
)


def test_recall_and_precision():
    got = ["a", "b", "c", "d"]
    rel = {"a", "d", "z"}
    assert recall_at_k(got, rel, 4) == pytest.approx(2 / 3)
    assert recall_at_k(got, rel, 1) == pytest.approx(1 / 3)
    assert precision_at_k(got, rel, 4) == pytest.approx(2 / 4)
    assert precision_at_k(got, rel, 2) == pytest.approx(1 / 2)


def test_recall_with_no_relevant_docs_is_zero_not_a_crash():
    assert recall_at_k(["a"], set(), 5) == 0.0


def test_mrr_uses_the_first_hit():
    assert mrr_at_k(["x", "y", "a"], {"a", "y"}, 10) == pytest.approx(1 / 2)
    assert mrr_at_k(["x", "y"], {"a"}, 10) == 0.0


def test_hit_at_k():
    assert hit_at_k(["x", "a"], {"a"}, 2) == 1.0
    assert hit_at_k(["x", "a"], {"a"}, 1) == 0.0


def test_ndcg_perfect_ranking_is_one():
    graded = {"a": 2, "b": 1}
    assert ndcg_at_k(["a", "b", "z"], graded, 10) == pytest.approx(1.0)


def test_ndcg_penalizes_a_worse_order():
    graded = {"a": 2, "b": 1}
    good = ndcg_at_k(["a", "b"], graded, 10)
    bad = ndcg_at_k(["b", "a"], graded, 10)
    assert bad < good


def test_ndcg_matches_hand_computation():
    # one relevant doc with grade 1 at rank 2: dcg = 1/log2(3), idcg = 1/log2(2)
    got = ndcg_at_k(["x", "a"], {"a": 1}, 10)
    assert got == pytest.approx((1 / math.log2(3)) / (1 / math.log2(2)))


def test_ndcg_ignores_docs_beyond_k():
    assert ndcg_at_k(["x", "y", "a"], {"a": 1}, 2) == 0.0


def test_squad_normalization():
    assert normalize_answer("The  Cat's, hat.") == "cats hat"
    assert normalize_answer("A dog") == "dog"


def test_exact_match_is_normalization_insensitive():
    assert exact_match("The Answer", ["answer"]) == 1.0
    assert exact_match("wrong", ["answer"]) == 0.0


def test_token_f1_partial_credit():
    # 2 of 3 predicted tokens are right, both gold tokens recovered
    assert token_f1("quick brown fox", ["brown fox"]) == pytest.approx(
        2 * (2 / 3) * 1.0 / (2 / 3 + 1.0)
    )
    assert token_f1("nothing alike", ["brown fox"]) == 0.0


def test_token_f1_ignores_articles():
    """SQuAD normalization strips a/an/the, so these are the same answer."""
    assert token_f1("the brown fox", ["brown fox"]) == pytest.approx(1.0)


def test_token_f1_takes_the_best_gold():
    assert token_f1("brown fox", ["totally different", "brown fox"]) == pytest.approx(1.0)


def test_faithfulness_flags_ungrounded_sentences():
    ctx = ["The deploy failed because the container ran out of memory."]
    grounded, bad = faithfulness("The deploy failed because the container ran out of memory.", ctx)
    assert grounded == pytest.approx(1.0) and not bad

    grounded, bad = faithfulness("The database was deleted by a rogue cron job.", ctx)
    assert grounded == 0.0 and len(bad) == 1


def test_faithfulness_on_an_empty_answer_is_vacuously_true():
    assert faithfulness("", ["anything"])[0] == 1.0


def test_roc_auc_perfect_and_random():
    assert roc_auc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == pytest.approx(1.0)
    assert roc_auc([0.9, 0.8, 0.2, 0.1], [False, False, True, True]) == pytest.approx(0.0)
    # every score identical carries no information at all
    assert roc_auc([0.5] * 4, [True, False, True, False]) == pytest.approx(0.5)


def test_roc_auc_handles_one_sided_labels():
    assert roc_auc([0.1, 0.9], [True, True]) == 0.5


def test_abstention_curve_endpoints():
    conf = [0.1, 0.4, 0.6, 0.9]
    ans = [False, False, True, True]
    curve = abstention_curve(conf, ans, steps=5)
    # lowest threshold answers everything: every unanswerable is a hallucination
    assert curve[0].hallucination_rate == 1.0
    assert curve[0].over_abstention_rate == 0.0
    # highest threshold abstains on nearly everything
    assert curve[-1].over_abstention_rate > 0


def test_pick_threshold_respects_the_safety_ceiling():
    conf = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    ans = [False, False, False, True, True, True]
    curve = abstention_curve(conf, ans, steps=21)
    t = pick_threshold(curve, "safe", max_hallucination=0.0)
    chosen = min((p for p in curve if p.threshold >= t), key=lambda p: p.threshold)
    assert chosen.hallucination_rate == 0.0
