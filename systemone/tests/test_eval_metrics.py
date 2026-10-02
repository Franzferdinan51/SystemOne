"""Tests for the eval-honesty metrics batch.

Rank metrics (rank-plans/rerank evals), failure AUROC, Brier
decomposition, reliability curves, paired A/B significance, and
label-free monitoring (PSI, CBPE estimate). All values hand-checked;
no model needed.
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from systemone.metrics import (  # noqa: E402
    brier_decomposition,
    estimated_accuracy,
    failure_auroc,
    macro_f1,
    mcnemar,
    ndcg_at_k,
    paired_bootstrap_ci,
    psi,
    reciprocal_rank,
    reliability_curve,
)


def test_ndcg_at_k():
    assert ndcg_at_k([3, 2, 1], k=3) == pytest.approx(1.0)
    assert ndcg_at_k([0, 0, 1], k=3) == pytest.approx(0.5)
    assert ndcg_at_k([1, 0, 1], k=3) == pytest.approx(
        1.5 / (1.0 + 1.0 / math.log2(3)))
    assert ndcg_at_k([1, 1], k=5) == pytest.approx(1.0)
    assert ndcg_at_k([0, 0, 0], k=3) == pytest.approx(1.0)
    assert ndcg_at_k([], k=3) == pytest.approx(0.0)
    assert ndcg_at_k([1, 0], k=0) == pytest.approx(0.0)


def test_reciprocal_rank():
    assert reciprocal_rank([0, 0, 1]) == pytest.approx(1 / 3)
    assert reciprocal_rank([1, 0]) == pytest.approx(1.0)
    assert reciprocal_rank([0, 0]) == pytest.approx(0.0)
    assert reciprocal_rank([]) == pytest.approx(0.0)


def test_macro_f1():
    assert macro_f1([0, 0, 1, 1], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert macro_f1([0, 0, 1, 1], [0, 1, 1, 1]) == pytest.approx(
        (2 / 3 + 0.8) / 2)
    assert macro_f1([0, 1], [1, 0]) == pytest.approx(0.0)
    # unseen label with no members either side scores 1.0
    assert macro_f1([0], [0], labels=[0, 1]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        macro_f1([], [])
    with pytest.raises(ValueError):
        macro_f1([0], [0, 1])


def test_failure_auroc():
    good = [[0.9, 0.1], [0.8, 0.2]]
    bad = [[0.55, 0.45], [0.4, 0.6]]
    assert failure_auroc([0, 0, 1, 1], good + bad) == pytest.approx(1.0)
    assert failure_auroc([1, 1, 0, 0], good + bad) == pytest.approx(0.0)
    assert failure_auroc([0, 0], good) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        failure_auroc([], [])


def test_brier_decomposition_identity():
    y = [0, 1, 0, 1, 0, 1, 0, 1]
    p = [[0.9, 0.1], [0.2, 0.8]] * 4
    d = brier_decomposition(y, p, bins=5)
    assert d["brier"] == pytest.approx(
        d["reliability"] - d["resolution"] + d["uncertainty"])
    assert d["reliability"] < 0.05
    assert d["uncertainty"] == pytest.approx(0.25)
    # confident-and-wrong inflates reliability, not uncertainty
    p2 = [[0.1, 0.9], [0.9, 0.1]] * 4
    d2 = brier_decomposition(y, p2, bins=5)
    assert d2["reliability"] > 0.5
    assert d2["uncertainty"] == pytest.approx(0.25)


def test_reliability_curve_counts():
    y = [0, 1, 0, 1]
    p = [[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]]
    curve = reliability_curve(y, p, bins=5)
    assert len(curve) == 5
    assert sum(b["count"] for b in curve) == pytest.approx(4.0)
    assert all(0.0 <= b["accuracy"] <= 1.0 for b in curve)


def test_paired_bootstrap_ci():
    a = [True] * 50 + [False] * 50
    b = [True] * 80 + [False] * 20
    ci = paired_bootstrap_ci(a, b, n_boot=500, seed=7)
    assert ci["delta"] == pytest.approx(0.3)
    assert ci["lo"] > 0.0  # B strictly better on shared items
    assert ci["lo"] < ci["hi"]
    same = paired_bootstrap_ci(a, a, n_boot=500, seed=7)
    assert same["delta"] == pytest.approx(0.0)
    assert same["lo"] <= 0.0 <= same["hi"]
    again = paired_bootstrap_ci(a, b, n_boot=500, seed=7)
    assert again == ci  # seeded => reproducible
    with pytest.raises(ValueError):
        paired_bootstrap_ci([], [])
    with pytest.raises(ValueError):
        paired_bootstrap_ci([True], [True, False])


def test_mcnemar():
    # b=10, c=0, exact branch: p = 2/1024
    a = [False] * 10 + [True] * 10
    b = [True] * 10 + [True] * 10
    r = mcnemar(a, b)
    assert r["b"] == 10.0 and r["c"] == 0.0
    assert r["p_value"] == pytest.approx(2 / 1024)
    # symmetric disagreements => p = 1
    a2 = [True, False] * 5
    b2 = [False, True] * 5
    assert mcnemar(a2, b2)["p_value"] == pytest.approx(1.0)
    assert mcnemar(a2, a2)["p_value"] == pytest.approx(1.0)
    # large-n chi-square branch: b=30, c=10
    a3 = [False] * 30 + [True] * 10 + [True] * 60
    b3 = [True] * 30 + [False] * 10 + [True] * 60
    r3 = mcnemar(a3, b3)
    assert r3["statistic"] == pytest.approx((20 - 1) ** 2 / 40)
    assert r3["p_value"] < 0.01
    with pytest.raises(ValueError):
        mcnemar([], [])


def test_psi():
    ref = [float(i) / 100 for i in range(100)]
    assert psi(ref, list(ref)) == pytest.approx(0.0)
    shifted = [x + 5.0 for x in ref]
    assert psi(ref, shifted) > 0.2
    assert psi([0.5] * 50, [0.5] * 50) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        psi([], [1.0])


def test_estimated_accuracy():
    p = [[0.9, 0.1], [0.6, 0.4], [0.2, 0.8]]
    assert estimated_accuracy(p) == pytest.approx((0.9 + 0.6 + 0.8) / 3)
    with pytest.raises(ValueError):
        estimated_accuracy([])
