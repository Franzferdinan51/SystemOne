"""Tests for conformal prediction sets and RouteLLM-style thresholds.

Split-conformal APS thresholds (MAPIE method, numpy only) and the
offline α picker that holds a quality target at minimum cost. All
values hand-checked; no model needed.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from systemone.calibration import (  # noqa: E402
    conformal_set,
    fit_conformal_threshold,
    quality_cost_frontier,
    route_threshold_for_target,
)


def _row(probs, gold, qtype="choice"):
    return {"type": qtype, "gold": gold, "probs": list(probs)}


def test_conformal_threshold_and_sets():
    rows = [_row([0.7, 0.2, 0.1], 0)] * 9 + [_row([0.5, 0.4, 0.1], 1)]
    fit = fit_conformal_threshold(rows, alpha=0.1)
    assert fit["n"] == 10
    assert fit["tau"] == pytest.approx(0.9)  # worst APS score at 90% level
    assert conformal_set({"a": 0.7, "b": 0.2, "c": 0.1}, 0.85) == ["a", "b"]
    assert conformal_set({"a": 0.7, "b": 0.2, "c": 0.1}, fit["tau"]) == [
        "a", "b", "c"]  # 0.7 + 0.2 < 0.9 in floats, so c joins
    assert conformal_set({"a": 0.7, "b": 0.2, "c": 0.1}, 0.7) == ["a"]
    assert conformal_set({"a": 0.5, "b": 0.5}, 0.0) == ["a"]  # top-1 always in
    # looser alpha => smaller tau on the same rows
    assert fit_conformal_threshold(rows, alpha=0.5)["tau"] == pytest.approx(0.7)


def test_conformal_empirical_coverage():
    rng = np.random.default_rng(11)
    rows = []
    for _ in range(400):
        k = int(rng.integers(2, 5))
        logits = rng.normal(size=k)
        p = np.exp(logits) / np.exp(logits).sum()
        gold = int(rng.choice(k, p=p))  # well-specified: draw from p
        rows.append(_row(p.tolist(), gold))
    tau = fit_conformal_threshold(rows[:200], alpha=0.1)["tau"]
    covered = 0
    for r in rows[200:]:
        dist = {f"o{i}": p for i, p in enumerate(r["probs"])}
        if f"o{r['gold']}" in conformal_set(dist, tau):
            covered += 1
    assert covered / 200 >= 0.85  # guarantee is 0.90; margin for the draw


def test_conformal_validation():
    with pytest.raises(ValueError):
        fit_conformal_threshold([_row([0.5, 0.5], 0)], alpha=0.0)
    with pytest.raises(ValueError):
        fit_conformal_threshold([_row([0.5, 0.5], 0)], alpha=1.0)
    with pytest.raises(ValueError):
        fit_conformal_threshold([_row([0.5, 0.5], 0, qtype="noul")])
    with pytest.raises(ValueError):
        conformal_set({}, 0.9)
    # logits rows work too
    fit = fit_conformal_threshold(
        [{"type": "choice", "gold": 1, "logits": [0.0, 2.0]}])
    assert fit["n"] == 1


def test_route_threshold_for_target():
    rows = [(0.1, True, True)] * 80 + [(0.9, False, True)] * 20
    r = route_threshold_for_target(rows, target=0.95, weak_cost=0.1,
                                   strong_cost=1.0)
    assert r["alpha"] == pytest.approx(0.9)
    assert r["routed_quality"] == pytest.approx(1.0)
    assert r["strong_quality"] == pytest.approx(1.0)
    assert r["weak_share"] == pytest.approx(0.8)
    assert r["cost_share"] == pytest.approx(0.28)
    # demanding full strong quality still finds the cheap α here
    r2 = route_threshold_for_target(rows, target=1.0)
    assert r2["alpha"] == pytest.approx(0.9)
    assert r2["routed_quality"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        route_threshold_for_target([], target=0.9)
    with pytest.raises(ValueError):
        route_threshold_for_target(rows, target=1.5)


def test_quality_cost_frontier():
    rows = [(0.1, True, True)] * 80 + [(0.9, False, True)] * 20
    pts = quality_cost_frontier(rows, weak_cost=0.1, strong_cost=1.0, points=3)
    assert [p["alpha"] for p in pts] == [0.0, 0.5, 1.0]
    assert [p["quality"] for p in pts] == pytest.approx([1.0, 1.0, 0.8])
    assert [p["weak_share"] for p in pts] == pytest.approx([0.0, 0.8, 1.0])
    assert pts[1]["cost_share"] == pytest.approx(0.28)
    with pytest.raises(ValueError):
        quality_cost_frontier([])
