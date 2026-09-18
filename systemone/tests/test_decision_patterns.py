"""Fast tests for the decision patterns ported from jev-ultrafast / mobile-jev.

No model needed — these exercise the pure response-contract helpers.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from systemone import (
    ABSTAIN_LABEL,
    LatencyStats,
    StallGuard,
    SystemOneError,
    validate_choice,
    validate_distribution,
    with_abstain,
)


def _good():
    return {
        "choice": "b",
        "probabilities": {"a": 0.2, "b": 0.5, "c": 0.3},
        "confidence": 0.5,
    }


def test_validate_choice_accepts_good():
    assert validate_choice(_good(), ["a", "b", "c"]) == _good()


def test_validate_choice_rejects_unknown_choice():
    bad = {**_good(), "choice": "zzz"}
    with pytest.raises(SystemOneError):
        validate_choice(bad, ["a", "b", "c"])


def test_validate_choice_rejects_key_mismatch():
    bad = {**_good(), "probabilities": {"a": 0.5, "b": 0.5}}
    with pytest.raises(SystemOneError):
        validate_choice(bad, ["a", "b", "c"])


def test_validate_choice_rejects_bad_sum():
    bad = {**_good(), "probabilities": {"a": 0.2, "b": 0.5, "c": 0.9}}
    with pytest.raises(SystemOneError):
        validate_choice(bad, ["a", "b", "c"])


def test_validate_choice_rejects_choice_not_max():
    bad = {**_good(), "choice": "a"}  # b holds the max
    with pytest.raises(SystemOneError):
        validate_choice(bad, ["a", "b", "c"])


def test_validate_choice_rejects_nonfinite():
    bad = {**_good(), "probabilities": {"a": 0.2, "b": float("nan"), "c": 0.8}}
    with pytest.raises(SystemOneError):
        validate_choice(bad, ["a", "b", "c"])


def test_validate_choice_rejects_malformed():
    with pytest.raises(SystemOneError):
        validate_choice({"nope": 1}, ["a"])


def test_with_abstain_appends_once():
    assert with_abstain(["x", "y"]) == ["x", "y", ABSTAIN_LABEL]
    assert with_abstain(["x", ABSTAIN_LABEL]).count(ABSTAIN_LABEL) == 1


def test_stall_guard_trips_after_three():
    g = StallGuard(max_stalls=3)
    assert g.observe(False) == "ok"
    assert g.observe(False) == "ok"
    assert g.observe(False) == "stalled"
    g.reset()
    assert g.observe(True) == "ok"


def test_stall_guard_progress_resets():
    g = StallGuard(max_stalls=2)
    g.observe(False)
    assert g.observe(True) == "ok"
    assert g.observe(False) == "ok"  # counter restarted


def test_latency_stats():
    s = LatencyStats()
    assert s.summary()["count"] == 0
    for ms in [10, 20, 30, 40, 100]:
        s.add(ms)
    out = s.summary()
    assert out["count"] == 5
    assert out["median_ms"] == 30
    assert out["p95_ms"] == 100
    assert out["mean_ms"] == 40.0
