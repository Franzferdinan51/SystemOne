"""Tests for judge fusion and sampling wrappers.

Rank fusion (RRF/Borda), extremized averaging, rank_models sort modes,
ConformalChoice prediction sets, SelfConsistent majority vote, and
EnsembleBackend strategies. Stub engines only; no model needed.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from systemone.rotation import (  # noqa: E402
    ConformalChoice,
    EnsembleBackend,
    SelfConsistent,
)
from systemone.scoring import (  # noqa: E402
    borda_fuse,
    extremized_average,
    mean_distributions,
    rank_models,
    rrf_fuse,
)


class Stub:
    def __init__(self, answers, name="stub"):
        self.answers = answers
        self.model_name = name
        self.calls = 0

    def systemone(self, state, questions, **kwargs):
        self.calls += 1
        return {k: (dict(v) if isinstance(v, dict) else v)
                for k, v in self.answers.items()}


def _choice(probs, name="q"):
    best = max(probs, key=probs.get)
    return {name: {"type": "choice", "choice": best,
                   "probabilities": dict(probs), "confidence": 0.5}}


def _questions():
    return [{"name": "q", "type": "choice",
             "options": ["a", "b", "c"]}]


def test_rrf_fuse_rank_order_and_k():
    fused = rrf_fuse([["a", "b", "c"], ["b", "a", "c"]])
    assert fused["a"] == pytest.approx(fused["b"])
    assert fused["a"] > fused["c"]
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert rrf_fuse([["a", "b"]])["a"] == pytest.approx(1 / 61)
    with pytest.raises(ValueError):
        rrf_fuse([])


def test_borda_fuse_points():
    fused = borda_fuse([["a", "b", "c"], ["b", "a", "c"]])
    assert fused == {"a": 1.5, "b": 1.5, "c": 0.0}
    assert borda_fuse([["x"]]) == {"x": 0.0}
    with pytest.raises(ValueError):
        borda_fuse([])


def test_mean_and_extremized():
    ds = [{"a": 0.6, "b": 0.4}, {"a": 0.8, "b": 0.2}]
    assert mean_distributions(ds) == pytest.approx({"a": 0.7, "b": 0.3})
    assert extremized_average(ds, strength=0.0) == pytest.approx({"a": 0.7, "b": 0.3})
    ext = extremized_average(ds, strength=1.0)
    assert ext["a"] > 0.7 and ext["b"] < 0.3
    assert sum(ext.values()) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        mean_distributions([])
    with pytest.raises(ValueError):
        extremized_average(ds, strength=-1.0)


def _registry():
    return {"tiers": {
        "economy": {"models": [
            {"model_id": "cheap", "available": True, "cost": 0.2,
             "quality": {"economy": 0.6, "heavy": 0.1},
             "latency_ms_p50": 100},
            {"model_id": "slow", "available": True, "cost": 0.9,
             "quality": {"economy": 0.9, "heavy": 0.9},
             "latency_ms_p50": 900},
            {"model_id": "nolat", "available": True, "cost": 0.5,
             "quality": {"economy": 0.7, "heavy": 0.5}},
        ]}}}


def test_rank_models_sort_modes():
    probs = {"economy": 1.0}
    by_util = [m["model_id"] for m in rank_models(_registry(), probs, lam=0.0)]
    assert by_util[0] == "slow"  # best quality wins at lam=0
    by_cost = [m["model_id"] for m in rank_models(
        _registry(), probs, lam=0.0, sort="cost")]
    assert by_cost == ["cheap", "nolat", "slow"]
    by_lat = [m["model_id"] for m in rank_models(
        _registry(), probs, lam=0.0, sort="latency")]
    assert by_lat == ["cheap", "slow", "nolat"]  # missing latency sorts last
    by_qual = [m["model_id"] for m in rank_models(
        _registry(), probs, lam=0.0, sort="quality")]
    assert by_qual == ["slow", "nolat", "cheap"]
    with pytest.raises(ValueError):
        rank_models(_registry(), probs, sort="vibes")


def test_conformal_choice_wrapper():
    stub = Stub({**_choice({"a": 0.7, "b": 0.2, "c": 0.1}),
                 "_meta": {"model": "stub"}})
    wrapped = ConformalChoice(stub, tau=0.85, alpha=0.1)
    out = wrapped.systemone("s", _questions())
    assert out["q"]["choice"] == "a"
    assert out["q"]["prediction_set"] == ["a", "b"]
    assert out["q"]["coverage"] == pytest.approx(0.9)
    assert out["_meta"]["conformal"] == {"tau": 0.85, "coverage": 0.9}
    fitted = ConformalChoice.fit(
        stub, [{"type": "choice", "gold": 0, "probs": [0.7, 0.2, 0.1]}] * 10)
    assert fitted.tau == pytest.approx(0.7)
    with pytest.raises(ValueError):
        ConformalChoice(stub, tau=0.9, alpha=0.0)


def test_self_consistent_unanimous():
    stub = Stub(_choice({"a": 0.7, "b": 0.3}))
    out = SelfConsistent(stub, samples=3, seed=1).systemone("s", _questions())
    assert stub.calls == 3
    assert out["q"]["choice"] == "a"
    assert out["q"]["agreement"] == pytest.approx(1.0)
    assert out["_meta"]["samples"] == 3
    with pytest.raises(ValueError):
        SelfConsistent(stub, samples=1)


def test_self_consistent_split_vote():
    dists = [{"a": 0.8, "b": 0.2}, {"a": 0.2, "b": 0.8}, {"a": 0.6, "b": 0.4}]

    class Flip:
        model_name = "flip"

        def __init__(self):
            self.i = 0

        def systemone(self, state, questions, **kwargs):
            d = dists[self.i % len(dists)]
            self.i += 1
            return _choice(d)

    out = SelfConsistent(Flip(), samples=3).systemone("s", _questions())
    assert out["q"]["choice"] == "a"  # 2 of 3 votes
    assert out["q"]["agreement"] == pytest.approx(2 / 3, abs=1e-4)
    assert out["q"]["probabilities"]["a"] == pytest.approx(
        (0.8 + 0.2 + 0.6) / 3)


def test_ensemble_average_and_vote():
    e1 = Stub(_choice({"a": 0.8, "b": 0.2}), "e1")
    e2 = Stub(_choice({"a": 0.3, "b": 0.7}), "e2")
    e3 = Stub(_choice({"a": 0.4, "b": 0.6}), "e3")
    avg = EnsembleBackend([e1, e2, e3], strategy="average")
    out = avg.systemone("s", _questions())
    assert out["q"]["probabilities"] == pytest.approx({"a": 0.5, "b": 0.5})
    assert out["_meta"]["strategy"] == "average"
    vote = EnsembleBackend([e1, e2, e3], strategy="vote")
    out = vote.systemone("s", _questions())
    assert out["q"]["choice"] == "b"  # 2 of 3 engines
    assert out["q"]["agreement"] == pytest.approx(2 / 3, abs=1e-4)


def test_ensemble_rank_strategies_keep_honest_probs():
    e1 = Stub(_choice({"a": 0.9, "b": 0.1}), "e1")
    e2 = Stub(_choice({"a": 0.9, "b": 0.1}), "e2")
    for strategy in ("rrf", "borda"):
        out = EnsembleBackend([e1, e2], strategy=strategy).systemone(
            "s", _questions())
        assert out["q"]["choice"] == "a"
        assert out["q"]["probabilities"] == {"a": 0.9, "b": 0.1}
        assert out["q"]["fusion_scores"]["a"] > out["q"]["fusion_scores"]["b"]
    out = EnsembleBackend([e1, e2], strategy="extremized").systemone(
        "s", _questions())
    assert out["q"]["probabilities"]["a"] > 0.9


def test_ensemble_fuses_noul_and_score():
    e1 = Stub({"ok": {"type": "noul", "probability": 0.8, "answer": True,
                       "confidence": 0.6},
               "lvl": {"type": "score", "level": "hi",
                       "distribution": {"lo": 0.2, "hi": 0.8},
                       "score": 0.8, "confidence": 0.6,
                       "legend": {"lo": "lo", "hi": "hi"}}})
    e2 = Stub({"ok": {"type": "noul", "probability": 0.6, "answer": True,
                       "confidence": 0.2},
               "lvl": {"type": "score", "level": "hi",
                       "distribution": {"lo": 0.4, "hi": 0.6},
                       "score": 0.6, "confidence": 0.2,
                       "legend": {"lo": "lo", "hi": "hi"}}})
    out = EnsembleBackend([e1, e2]).systemone("s", [
        {"name": "ok", "type": "noul"},
        {"name": "lvl", "type": "score", "levels": ["lo", "hi"]}])
    assert out["ok"]["probability"] == pytest.approx(0.7)
    assert out["lvl"]["distribution"] == pytest.approx({"lo": 0.3, "hi": 0.7})
    assert out["lvl"]["level"] == "hi"


def test_ensemble_validation():
    e1 = Stub(_choice({"a": 1.0}))
    with pytest.raises(ValueError):
        EnsembleBackend([e1])
    with pytest.raises(ValueError):
        EnsembleBackend([e1, e1], strategy="lottery")
