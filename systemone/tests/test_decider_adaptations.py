"""Tests for the Mapika/decider cherry-picks.

Covers:
- calibration.PerTypeTemperatureCalibrator (per-answer-type temperature map,
  <50-rows pooled fallback, JSON round-trip, fail-open from_dict)
- patterns choice/score/noul TypeSafe confidence definitions (api re-exports)
- patterns.build_decision_prompts (state-first rows, option shuffling, abstain pin)
- metrics (ece/brier/nll/aurc/selective-accuracy/summarize/format_table)
- scoring.temperature_for + apply_calibration(qtype=...)
- battery.run.per_tier_stats + battery.fit_types merge behavior

No model needed — everything runs on synthetic data.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from systemone import patterns as api  # canonical light source; api re-exports
from systemone.patterns import (
    build_decision_prompts,
    choice_confidence,
    noul_confidence,
    score_confidence,
)
from systemone.calibration import (
    MIN_ROWS_PER_TYPE,
    PerTypeTemperatureCalibrator,
    fit_temperature_by_type,
    load_type_calibration,
)
from systemone.metrics import (
    aurc,
    brier_score,
    format_table,
    nll_score,
    selective_accuracy,
    summarize,
    top_label_ece,
)
from systemone.scoring import apply_calibration, temperature_for

BATTERY_DIR = os.path.join(os.path.dirname(__file__), "..", "battery")


# -- helpers ---------------------------------------------------------------

def _overconfident_rows(n, k, seed, correct_frac=0.7):
    """Synthetic T=1 simplex rows: peaked (overconfident), correct_frac right."""
    rng = np.random.RandomState(seed)
    logits = rng.randn(n, k) * 3.0
    gold = logits.argmax(axis=1)
    flip = rng.rand(n) > correct_frac
    gold[flip] = rng.randint(0, k, size=int(flip.sum()))
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    P = e / e.sum(axis=1, keepdims=True)
    return P, gold


def _records(P, gold, qtype, as_probs=True):
    recs = []
    for p, g in zip(P, gold):
        rec = {"type": qtype, "gold": int(g)}
        rec["probs" if as_probs else "logits"] = [float(x) for x in p]
        recs.append(rec)
    return recs


# -- per-type temperature ---------------------------------------------------

def test_per_type_fit_applies_own_temperature():
    P_c, y_c = _overconfident_rows(300, 4, seed=0)
    P_n, y_n = _overconfident_rows(300, 2, seed=1)
    records = _records(P_c, y_c, "choice") + _records(P_n, y_n, "noul")
    cal = fit_temperature_by_type(records)
    assert cal.fitted_
    assert cal.rows_by_type_["choice"] == 300
    assert cal.rows_by_type_["noul"] == 300
    assert cal.rows_by_type_["score"] == 0
    # overconfident synthetic data -> soften (T > 1)
    assert cal.temperature_for("choice") > 1.0
    assert cal.temperature_for("noul") > 1.0
    # score had no rows -> pooled fallback
    assert cal.temperature_for("score") == pytest.approx(cal.temperature_)
    assert "score" not in cal.temperature_by_type_
    # the map is actually applied per type
    scores = np.array([3.0, 0.5, 0.2, 0.1])
    p_choice = cal.predict_proba(scores, "choice")
    p_noul_fallback = cal.predict_proba(scores, "noul")
    assert p_choice.sum() == pytest.approx(1.0)
    assert not np.allclose(p_choice, p_noul_fallback)


def test_per_type_fallback_below_min_rows():
    P_c, y_c = _overconfident_rows(200, 3, seed=2)
    P_s, y_s = _overconfident_rows(MIN_ROWS_PER_TYPE - 1, 3, seed=3)
    records = _records(P_c, y_c, "choice") + _records(P_s, y_s, "score")
    cal = fit_temperature_by_type(records)
    assert cal.rows_by_type_["score"] == MIN_ROWS_PER_TYPE - 1
    assert "score" not in cal.temperature_by_type_
    assert cal.temperature_for("score") == pytest.approx(cal.temperature_)
    assert "choice" in cal.temperature_by_type_


def test_per_type_fit_accepts_logits_rows():
    P, y = _overconfident_rows(120, 3, seed=4)
    rng = np.random.RandomState(4)
    logits = rng.randn(120, 3) * 3.0
    records = [{"type": "choice", "gold": int(g), "logits": [float(x) for x in row]}
               for row, g in zip(logits, y)]
    cal = fit_temperature_by_type(records)
    assert cal.temperature_for("choice") > 0
    assert cal.predict_proba([1.0, 2.0, 3.0], "choice").sum() == pytest.approx(1.0)


def test_per_type_rejects_bad_records():
    cal = PerTypeTemperatureCalibrator()
    with pytest.raises(ValueError):
        cal.fit([])
    with pytest.raises(ValueError):
        cal.fit([{"type": "mystery", "gold": 0, "probs": [0.5, 0.5]}])
    with pytest.raises(ValueError):
        cal.fit([{"type": "choice", "gold": 0}])  # no logits/probs
    with pytest.raises(ValueError):
        cal.fit([{"type": "choice", "gold": 5, "probs": [0.5, 0.5]}])
    with pytest.raises(AssertionError):
        PerTypeTemperatureCalibrator().temperature_for("choice")


def test_per_type_dict_roundtrip_and_fail_open():
    P, y = _overconfident_rows(200, 3, seed=5)
    cal = fit_temperature_by_type(_records(P, y, "choice"))
    d = cal.to_dict()
    assert d["temperature"] == pytest.approx(cal.temperature_)
    assert d["min_rows_per_type"] == MIN_ROWS_PER_TYPE
    assert "note" in d and "fit_date" in d
    back = PerTypeTemperatureCalibrator.from_dict(json.loads(json.dumps(d)))
    assert back.temperature_for("choice") == pytest.approx(cal.temperature_for("choice"))
    assert back.temperature_for("noul") == pytest.approx(back.temperature_)
    # fail-open: insane per-type entries are dropped -> pooled fallback
    d["temperature_by_type"]["noul"] = -3.0
    d["temperature_by_type"]["score"] = "junk"
    evil = PerTypeTemperatureCalibrator.from_dict(d)
    assert evil.temperature_for("noul") == pytest.approx(evil.temperature_)
    assert evil.temperature_for("score") == pytest.approx(evil.temperature_)


def test_per_type_save_load_tmp(tmp_path):
    P, y = _overconfident_rows(200, 3, seed=6)
    cal = fit_temperature_by_type(_records(P, y, "choice"))
    path = str(tmp_path / "map.json")
    cal.save(path)
    back = PerTypeTemperatureCalibrator.load(path)
    assert back.temperature_for("choice") == pytest.approx(cal.temperature_for("choice"))


def test_load_type_calibration_none_when_absent(tmp_path):
    assert load_type_calibration(str(tmp_path / "missing.json")) is None
    pooled_only = tmp_path / "pooled.json"
    pooled_only.write_text(json.dumps({"temperature": 0.36}))
    assert load_type_calibration(str(pooled_only)) is None


# -- TypeSafe confidence ----------------------------------------------------

def test_choice_confidence_formula():
    # delta -> 1, uniform -> 0  ((n*p_max - 1) / (n - 1))
    assert choice_confidence([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert choice_confidence([1 / 3] * 3) == pytest.approx(0.0)
    assert choice_confidence([0.7, 0.2, 0.1]) == pytest.approx((3 * 0.7 - 1) / 2)
    assert choice_confidence(np.array([0.5, 0.5])) == pytest.approx(0.0)
    # degenerate single option -> p_max
    assert choice_confidence([1.0]) == pytest.approx(1.0)
    assert 0.0 <= choice_confidence([0.4, 0.35, 0.25]) <= 1.0


def test_score_confidence_formula():
    # all mass on one level -> 1; spread mass -> lower
    assert score_confidence([1.0, 0.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert score_confidence([0.0, 0.0, 0.0, 1.0]) == pytest.approx(1.0)
    split = score_confidence([0.5, 0.5, 0.0, 0.0])
    far = score_confidence([0.5, 0.0, 0.0, 0.5])
    assert 0.0 <= far < split < 1.0  # distance from the mode matters
    # two levels, k=argmax: 1 - p_far / (n-1)
    assert score_confidence([0.8, 0.2]) == pytest.approx(0.8)
    assert score_confidence([0.5, 0.5]) == pytest.approx(0.5)


def test_noul_confidence_is_chosen_probability():
    assert noul_confidence(0.72) == pytest.approx(0.72)
    assert noul_confidence(0.2) == pytest.approx(0.8)
    assert noul_confidence(0.5) == pytest.approx(0.5)


# -- prompt rows ------------------------------------------------------------

def _qs():
    return [
        {"name": "team", "type": "choice", "options": ["backend", "frontend", "devops"]},
        {"name": "urgency", "type": "score", "levels": ["low", "high"]},
        {"name": "page", "type": "noul", "statement": "Should we page?"},
    ]


def test_prompt_row_state_first_template():
    prompts, label_lists = build_decision_prompts("server is on fire", _qs())
    assert len(prompts) == 3 and len(label_lists) == 3
    row = prompts[0]
    assert row.startswith("Context:\nserver is on fire\n\nQuestion [1]:")
    assert "Options:\n(A) backend\n(B) frontend\n(C) devops" in row
    assert row.rstrip().endswith("Answer [1]: (")
    # score row lists levels in order, noul row embeds the statement
    assert "(A) low\n(B) high" in prompts[1]
    assert "Question [3]: Should we page?" in prompts[2]
    # label lists pass through unchanged without shuffling
    assert label_lists[0] == ["backend", "frontend", "devops"]


def test_prompt_row_shuffle_is_deterministic_with_seed():
    q = [{"name": "t", "type": "choice",
          "options": ["a", "b", "c", "d", "e"]}]
    p1, labs1 = build_decision_prompts("s", q, shuffle_options=True, seed=7)
    p2, labs2 = build_decision_prompts("s", q, shuffle_options=True, seed=7)
    assert labs1 == labs2 and p1 == p2
    p3, labs3 = build_decision_prompts("s", q, shuffle_options=True, seed=8)
    assert labs1 != labs3  # (overwhelmingly likely; same seed path is exact)
    # row text and label list agree
    for i, lab in enumerate(labs1[0]):
        assert f"({chr(65 + i)}) {lab}" in p1[0]


def test_prompt_row_shuffle_keeps_score_order_and_abstain_last():
    qs = [
        {"name": "u", "type": "score", "levels": ["low", "med", "high"]},
        {"name": "t", "type": "choice",
         "options": ["x", "y", api.ABSTAIN_LABEL, "z"]},
    ]
    _, labs = build_decision_prompts("s", qs, shuffle_options=True, seed=1)
    assert labs[0] == ["low", "med", "high"]  # ordered levels never shuffled
    assert labs[1][-1] == api.ABSTAIN_LABEL   # abstain pinned last
    assert sorted(labs[1][:-1]) == ["x", "y", "z"]


def test_prompt_row_empty_options_raises():
    with pytest.raises(api.SystemOneError):
        build_decision_prompts("s", [{"name": "t", "type": "choice", "options": []}])
    with pytest.raises(api.SystemOneError):
        build_decision_prompts("s", [{"name": "t", "type": "bogus"}])


# -- metrics ----------------------------------------------------------------

def test_top_label_ece_perfect_and_terrible():
    y = [0, 0, 1, 1, 2, 2]
    perfect = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
               [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
    assert top_label_ece(y, perfect) == pytest.approx(0.0)
    terrible = [[0.0, 0.5, 0.5], [0.0, 0.5, 0.5], [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.5, 0.0]]
    assert top_label_ece(y, terrible) > 0.5
    with pytest.raises(ValueError):
        top_label_ece([], [])


def test_brier_and_nll():
    y = [0, 1]
    P = [[1.0, 0.0], [0.0, 1.0]]
    assert brier_score(y, P) == pytest.approx(0.0)
    assert nll_score(y, P) == pytest.approx(0.0)
    P2 = [[0.5, 0.5], [0.5, 0.5]]
    assert brier_score(y, P2) == pytest.approx(0.5)
    assert nll_score(y, P2) == pytest.approx(-np.log(0.5))
    assert brier_score(y, P) < brier_score(y, P2)


def test_aurc_perfect_is_zero():
    y = [0, 0, 1, 1]
    perfect = [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]
    assert aurc(y, perfect) == pytest.approx(0.0)
    # anti-ranked: confident but wrong -> high AURC
    bad = [[0.1, 0.9], [0.2, 0.8], [0.9, 0.1], [0.8, 0.2]]
    assert aurc(y, bad) > aurc(y, perfect)


def test_selective_accuracy():
    y = [0, 0, 1, 1]
    P = [[0.99, 0.01], [0.02, 0.98], [0.98, 0.02], [0.01, 0.99]]
    # top-50% by confidence are the two 0.99 rows, both correct
    assert selective_accuracy(y, P, 0.5) == pytest.approx(1.0)
    assert selective_accuracy(y, P, 1.0) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        selective_accuracy(y, P, 0.0)


def test_summarize_and_format_table():
    y = [0, 0, 1, 1]
    P = [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]
    s = summarize(y, P, name="demo")
    assert s["name"] == "demo" and s["n"] == 4.0
    assert s["accuracy"] == pytest.approx(1.0)
    assert s["ece_15"] == pytest.approx(0.15, abs=0.01)
    for k in ("brier", "nll", "aurc", "sel_acc@50", "sel_acc@80", "sel_acc@100"):
        assert k in s and np.isfinite(s[k])
    table = format_table([s, summarize(y, P, name="demo2")])
    assert "ece15" in table and "demo" in table and "demo2" in table
    assert format_table([]) == "(no rows)"


# -- scoring.temperature_for --------------------------------------------------

def test_temperature_for_picks_per_type():
    cal = {"temperature": 0.5,
           "temperature_by_type": {"choice": 0.7, "noul": 1.3}}
    assert temperature_for(cal, "choice") == pytest.approx(0.7)
    assert temperature_for(cal, "noul") == pytest.approx(1.3)
    assert temperature_for(cal, "score") == pytest.approx(0.5)  # pooled fallback
    assert temperature_for({"temperature": 0.5}, "choice") == pytest.approx(0.5)
    assert temperature_for(None, "choice") == pytest.approx(1.0)
    assert temperature_for({"temperature": -2.0}, "choice") == pytest.approx(1.0)
    assert temperature_for({"temperature": 0.5,
                            "temperature_by_type": {"choice": 0.0}},
                           "choice") == pytest.approx(0.5)


def test_apply_calibration_uses_qtype():
    cal = {"temperature": 1.0,
           "temperature_by_type": {"choice": 2.0}}  # soften choice only
    probs = {"a": 0.9, "b": 0.1}
    out_choice = apply_calibration(probs, cal, qtype="choice")
    out_noul = apply_calibration(probs, cal, qtype="noul")
    assert out_choice["calibrated"] and out_noul["calibrated"]
    assert out_choice["calibrated_probabilities"]["a"] < 0.9  # softened
    assert out_noul["calibrated_probabilities"]["a"] == pytest.approx(0.9)  # T=1
    # default qtype keeps old behavior
    out_default = apply_calibration(probs, {"temperature": 2.0})
    assert out_default["calibrated_probabilities"]["a"] < 0.9


# -- battery per-tier stats ---------------------------------------------------

def test_per_tier_stats():
    sys.path.insert(0, BATTERY_DIR)
    import run as battery_run

    results = [
        {"true_tier": "economy", "pred_tier": "economy", "confidence": 0.9},
        {"true_tier": "economy", "pred_tier": "balanced", "confidence": 0.6},
        {"true_tier": "heavy", "pred_tier": "heavy", "confidence": 0.8},
    ]
    stats = battery_run.per_tier_stats(results)
    assert stats["economy"]["n"] == 2
    assert stats["economy"]["accuracy"] == pytest.approx(0.5)
    assert stats["heavy"]["accuracy"] == pytest.approx(1.0)
    for tier, s in stats.items():
        assert 0.0 <= s["ece"] <= 1.0
    assert battery_run.per_tier_stats([]) == {}


def test_fit_types_merges_into_calibration_json(tmp_path):
    sys.path.insert(0, BATTERY_DIR)
    import fit_types

    P, y = _overconfident_rows(120, 3, seed=9)
    recs = _records(P, y, "choice")
    rec_path = tmp_path / "records.jsonl"
    rec_path.write_text("\n".join(json.dumps(r) for r in recs))
    cal_path = tmp_path / "calibration.json"
    cal_path.write_text(json.dumps({"temperature": 0.362, "tiers": ["a", "b"]}))

    argv = ["fit_types.py", "--records", str(rec_path),
            "--calibration", str(cal_path), "--out", str(cal_path)]
    old = sys.argv
    sys.argv = argv
    try:
        assert fit_types.main() == 0
    finally:
        sys.argv = old

    merged = json.loads(cal_path.read_text())
    assert merged["temperature"] == pytest.approx(0.362)  # pooled untouched
    assert merged["tiers"] == ["a", "b"]
    assert "choice" in merged["temperature_by_type"]
    assert merged["temperature_by_type"]["choice"] > 1.0
    assert merged["per_type_rows"]["choice"] == 120


# -- calibrator JSON files (safe alternative to pickle) --------------------


def test_temperature_calibrator_json_round_trip(tmp_path):
    from systemone.calibration import TemperatureCalibrator

    cal = TemperatureCalibrator()
    cal.temperature_, cal.fitted_ = 1.7, True
    path = str(tmp_path / "cal.json")
    assert cal.save(path) == path
    loaded = TemperatureCalibrator.load(path)
    assert loaded.temperature_ == pytest.approx(1.7)
    assert loaded.fitted_ is True


def test_temperature_calibrator_from_dict_fails_open():
    from systemone.calibration import TemperatureCalibrator

    assert TemperatureCalibrator.from_dict({}).temperature_ == 1.0
    assert TemperatureCalibrator.from_dict({"temperature": -3}).temperature_ == 1.0
    assert TemperatureCalibrator.from_dict({"temperature": "junk"}).temperature_ == 1.0


def test_load_calibrator_file_dispatches_by_kind(tmp_path):
    import pickle

    from systemone.calibration import (
        PerTypeTemperatureCalibrator,
        TemperatureCalibrator,
        load_calibrator_file,
    )

    pooled = tmp_path / "pooled.json"
    pooled.write_text(json.dumps({"temperature": 2.0}))
    got = load_calibrator_file(str(pooled))
    assert isinstance(got, TemperatureCalibrator)
    assert got.temperature_ == pytest.approx(2.0)

    per_type = tmp_path / "types.json"
    per_type.write_text(json.dumps({
        "temperature": 1.0, "temperature_by_type": {"choice": 1.5},
    }))
    got = load_calibrator_file(str(per_type))
    assert isinstance(got, PerTypeTemperatureCalibrator)
    assert got.temperature_for("choice") == pytest.approx(1.5)

    legacy = tmp_path / "legacy.pkl"
    legacy.write_bytes(pickle.dumps({"not": "a calibrator"}))
    with pytest.warns(DeprecationWarning, match="re-save as JSON"):
        got = load_calibrator_file(str(legacy))
    assert got == {"not": "a calibrator"}  # legacy path preserves behavior
