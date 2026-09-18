"""Smoke tests for systemone. Model-dependent tests are marked slow.

Run fast tests only:  pytest systemone/tests -m "not slow"
Run all (downloads/loads gliclass-edge once):  pytest systemone/tests
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from systemone.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    TemperatureCalibrator,
    expected_calibration_error,
    softmax,
)


def test_softmax_rows_sum_to_one():
    P = softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(P.sum(axis=1), 1.0)


def test_ece_perfect_is_zero():
    y = [1, 1, 0, 0]
    p = [1.0, 1.0, 0.0, 0.0]
    assert expected_calibration_error(y, p, n_bins=2) == pytest.approx(0.0)


def test_ece_terrible_is_high():
    y = [1, 1, 0, 0]
    p = [0.0, 0.0, 1.0, 1.0]
    assert expected_calibration_error(y, p, n_bins=2) > 0.9


def test_temperature_fit_reduces_nll():
    rng = np.random.RandomState(0)
    # overconfident synthetic scores: big margins, 80% correct
    n, k = 400, 3
    S = rng.randn(n, k) * 4.0
    y = S.argmax(axis=1)
    flip = rng.rand(n) < 0.2
    y[flip] = rng.randint(0, k, size=flip.sum())

    def nll(P, yy):
        return -np.log(np.clip(P[np.arange(n), yy], 1e-12, 1)).mean()

    before = nll(softmax(S), y)
    cal = TemperatureCalibrator().fit(S, y)
    after = nll(cal.predict_proba(S), y)
    assert cal.temperature_ > 1.0  # overconfident -> soften
    assert after < before


def test_platt_binary():
    rng = np.random.RandomState(1)
    s = rng.rand(300)
    y = (s > 0.5).astype(int)
    cal = PlattCalibrator().fit(s, y)
    p = cal.predict_proba([0.9, 0.1])
    assert p[0] > 0.7 and p[1] < 0.3

def test_isotonic_binary_monotone():
    rng = np.random.RandomState(2)
    s = rng.rand(400)
    y = (s + rng.randn(400) * 0.2 > 0.5).astype(int)
    cal = IsotonicCalibrator().fit(s, y)
    p = cal.predict_proba([0.1, 0.5, 0.9])
    assert p[0] <= p[1] <= p[2]


@pytest.mark.slow
def test_systemone_end_to_end():
    from systemone import SystemOne, make_questions

    eng = SystemOne()  # loads one tiny model
    out = eng.systemone(
        "The server room is on fire",
        make_questions(
            choices={"team": ["backend", "devops"]},
            scores={"urgency": ["low", "high"]},
            nouls={"evacuate": "Should everyone evacuate the building?"},
        ),
    )
    assert out["team"]["choice"] in ("backend", "devops")
    assert abs(sum(out["team"]["probabilities"].values()) - 1.0) < 1e-6
    assert out["urgency"]["level"] in ("low", "high")
    assert 0.0 <= out["evacuate"]["probability"] <= 1.0
    assert out["_meta"]["n_questions"] == 3


# -- Loki cherry-picks: sanitized errors, state bound, confidence fallback ---

def test_systemone_error_is_sanitized_and_hinted():
    from systemone import SystemOneError

    err = SystemOneError("could not load any GLiClass model",
                         hint="last error: OSError: /secret/path/key boom")
    text = str(err)
    assert "could not load any GLiClass model" in text
    # message stays one line and carries the actionable hint
    assert "\n" not in text
    assert err.hint.startswith("last error:")


def test_sanitize_detail_is_single_line_and_bounded():
    from systemone.api import _sanitize_detail

    err = ValueError("line one\nline two\n" + "x" * 500)
    s = _sanitize_detail(err)
    assert "\n" not in s
    assert len(s) <= 300
    assert s.startswith("ValueError:")


def test_max_state_chars_bound():
    from systemone import MAX_STATE_CHARS

    assert MAX_STATE_CHARS == 6000  # same bound Loki uses for the routing task


def test_mcp_confidence_fallback():
    from systemone.mcp_server import _confidence

    # missing confidence -> max of the distribution (Loki's _extract_route_answer)
    assert _confidence({}, {"a": 0.2, "b": 0.8}) == 0.8
    # present confidence wins
    assert _confidence({"confidence": 0.42}, {"a": 0.2, "b": 0.8}) == 0.42
    # nothing at all -> 0.0, never crashes
    assert _confidence({}, {}) == 0.0


def test_autorouter_threshold_clamped():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "demo_autorouter",
        os.path.join(os.path.dirname(__file__), "..", "examples", "demo_autorouter.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod._clamp_threshold(2.5) == 1.0
    assert mod._clamp_threshold(-1.0) == 0.0
    assert mod._clamp_threshold(0.7) == 0.7
    assert mod._clamp_threshold("junk") == mod.CONFIDENCE_THRESHOLD


def test_autorouter_candidate_descriptions_structured():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "demo_autorouter2",
        os.path.join(os.path.dirname(__file__), "..", "examples", "demo_autorouter.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Loki-style structured format: capability tokens + context + cost tier
    for desc in mod.CANDIDATE_DESCRIPTIONS.values():
        assert "context=" in desc
        assert "cost=" in desc


def test_autorouter_cache_bounded(tmp_path, monkeypatch):
    import importlib.util
    import json

    spec = importlib.util.spec_from_file_location(
        "demo_autorouter3",
        os.path.join(os.path.dirname(__file__), "..", "examples", "demo_autorouter.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cache_file = tmp_path / "cache.json"
    monkeypatch.setattr(mod, "ROUTE_CACHE", str(cache_file))
    monkeypatch.setattr(mod, "ROUTE_CACHE_MAX_ENTRIES", 3)
    for i in range(5):
        mod._save_sticky(f"fp{i}", {"chosen_model": f"m{i}"})
    data = json.loads(cache_file.read_text())
    assert len(data) == 3
    # oldest evicted, newest kept
    assert "fp4" in data and "fp0" not in data
