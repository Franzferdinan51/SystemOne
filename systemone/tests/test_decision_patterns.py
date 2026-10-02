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


# -- security helpers (patterns.require_http_url / resolve_revision) -------


def test_require_http_url_accepts_http_and_https():
    from systemone.patterns import require_http_url

    assert require_http_url("http://127.0.0.1:8765") == "http://127.0.0.1:8765"
    assert require_http_url("  https://gpu-box:30000/x ") == "https://gpu-box:30000/x"


def test_require_http_url_rejects_non_http():
    from systemone.patterns import require_http_url

    for bad in ("file:///etc/passwd", "ftp://x/y", "gopher://x", "",
                "///no-scheme", "notaurl"):
        try:
            require_http_url(bad, what="test URL")
        except ValueError as exc:
            assert "test URL" in str(exc)
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")


def test_resolve_revision_precedence(monkeypatch):
    from systemone.patterns import resolve_revision

    monkeypatch.delenv("SYSTEMONE_REVISION", raising=False)
    assert resolve_revision(None) is None
    assert resolve_revision("  abc123 ") == "abc123"
    monkeypatch.setenv("SYSTEMONE_REVISION", "env-rev")
    assert resolve_revision(None) == "env-rev"
    assert resolve_revision("explicit") == "explicit"  # explicit wins


# -- date_facts (ported from Kev) -------------------------------------------


def test_date_facts_pairs_absolute_dates():
    from systemone.patterns import date_facts

    out = date_facts("Signed July 22, 2026, delivered August 3, 2026.")
    assert out == "August 3, 2026 is 12 days after July 22, 2026."
    iso = date_facts("from 2026-01-01 to 2026-01-02")
    assert iso == "2026-01-02 is 1 day after 2026-01-01."


def test_date_facts_needs_two_dates():
    from systemone.patterns import date_facts

    assert date_facts("Signed July 22, 2026.") == ""
    assert date_facts("no dates here") == ""
    assert date_facts("") == ""


def test_date_facts_same_day_and_duplicates():
    from systemone.patterns import date_facts

    assert date_facts("2026-01-01 then January 1, 2026") == (
        "January 1, 2026 is the same day as 2026-01-01.")
    # repeated identical strings count once -> no pairs
    assert date_facts("2026-01-01 and 2026-01-01") == ""


def test_with_date_facts_shapes():
    from systemone.patterns import with_date_facts

    two = "July 22, 2026 to August 3, 2026"
    assert with_date_facts("nothing dated") == "nothing dated"
    assert "date_facts: " in with_date_facts(two)
    d = with_date_facts({"state": two})
    assert "date_facts" in d and d["state"] == two
    lst = with_date_facts([two])
    assert lst[-1] == {"date_facts": lst[-1]["date_facts"]}


def test_shim_date_facts_opt_in(monkeypatch):
    from systemone.shim import translate_body

    body = {"state": "Signed July 22, 2026, delivered August 3, 2026.",
            "questions": {"q": {"type": "noul", "instructions": "late?"}}}
    monkeypatch.delenv("SYSTEMONE_DATE_FACTS", raising=False)
    state, _ = translate_body(body)
    assert "date_facts" not in state
    monkeypatch.setenv("SYSTEMONE_DATE_FACTS", "1")
    state, _ = translate_body(body)
    assert "date_facts: August 3, 2026 is 12 days after" in state
