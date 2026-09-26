"""Tests for the SYSTEMONE_DECISION_BACKEND switch (jeff1 | decider).

Covers (no model weights needed — Decider is mocked throughout):

- backend selection via env: decider -> DeciderEngine, jeff1/unset ->
  Jeff1Engine (the rollback default), unknown value -> ValueError
- DeciderEngine adapter mapping for choice / noul / score against the
  decider-ai answer shapes (answer["choice"] + answer["probabilities"],
  answer["noul"], score argmax over answer["probabilities"])
- score prediction is argmax over the level probabilities, never the
  rounded "score" expectation field
- Jeff1Engine.score delegates to its choice readout (behavior unchanged)
- end-to-end: POST /v1/jeff1/decide through the real handler with a
  mocked Decider
"""

import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import systemone.jeff1_sidecar as mod  # noqa: E402
from systemone.jeff1_sidecar import (  # noqa: E402
    DeciderEngine,
    Jeff1Engine,
    Jeff1Handler,
)


# -- fake decider -------------------------------------------------------------


class _FakeDecider:
    """Stands in for decider.infer.Decider: canned system_one answers."""

    name = "decider-4b-v2.1"
    layout = "plain"
    isolated_levels = True
    T_by_type = {"choice": 1.11, "noul": 1.56, "score": 1.287}

    def __init__(self, answers):
        self.answers = answers  # qtype -> answer dict
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        qtype = questions["q"]["type"]
        return {"answers": {"q": dict(self.answers[qtype])}}


def _engine_with_fake(answers):
    """DeciderEngine with load() bypassed: tests the adapter mapping only."""
    eng = DeciderEngine()
    eng._decider = _FakeDecider(answers)
    return eng


@pytest.fixture()
def clean_engine():
    """Isolate get_engine()'s module-global cache + the env switch."""
    saved_engine, saved_env = mod._ENGINE, os.environ.get(
        "SYSTEMONE_DECISION_BACKEND")
    mod._ENGINE = None
    try:
        yield
    finally:
        mod._ENGINE = saved_engine
        if saved_env is None:
            os.environ.pop("SYSTEMONE_DECISION_BACKEND", None)
        else:
            os.environ["SYSTEMONE_DECISION_BACKEND"] = saved_env


# -- backend selection ----------------------------------------------------------


def test_backend_defaults_to_jeff1(clean_engine, monkeypatch):
    monkeypatch.delenv("SYSTEMONE_DECISION_BACKEND", raising=False)
    assert isinstance(mod.get_engine(), Jeff1Engine)


def test_backend_jeff1_explicit(clean_engine, monkeypatch):
    monkeypatch.setenv("SYSTEMONE_DECISION_BACKEND", "jeff1")
    assert isinstance(mod.get_engine(), Jeff1Engine)


def test_backend_decider(clean_engine, monkeypatch):
    monkeypatch.setenv("SYSTEMONE_DECISION_BACKEND", "decider")
    eng = mod.get_engine()
    assert isinstance(eng, DeciderEngine)
    assert "decider-4b" in eng.model_id


def test_backend_value_is_case_insensitive(clean_engine, monkeypatch):
    monkeypatch.setenv("SYSTEMONE_DECISION_BACKEND", "Decider")
    assert isinstance(mod.get_engine(), DeciderEngine)


def test_backend_unknown_raises(clean_engine, monkeypatch):
    monkeypatch.setenv("SYSTEMONE_DECISION_BACKEND", "bogus")
    with pytest.raises(ValueError, match="unknown SYSTEMONE_DECISION_BACKEND"):
        mod.get_engine()


def test_decider_pin_defaults(clean_engine, monkeypatch):
    monkeypatch.delenv("DECIDER_REPO_ID", raising=False)
    monkeypatch.delenv("DECIDER_REVISION", raising=False)
    assert mod.decider_repo_id() == "Mapika/decider-4b"
    assert mod.decider_revision() == \
        "eb5fbdfc9448473ec25e399882912863afbdb70e"


# -- adapter mapping --------------------------------------------------------------


def test_choice_mapping():
    eng = _engine_with_fake({
        "choice": {"type": "choice", "choice": "b",
                   "probabilities": {"a": 0.2, "b": 0.8},
                   "x_p_max": 0.8},
    })
    label, probs, conf = eng.choice(
        {"claim": "x"}, "Pick one.", {"a": "first", "b": "second"})
    assert label == "b"
    assert probs == {"a": 0.2, "b": 0.8}
    assert conf == pytest.approx(0.8)
    state, questions = eng._decider.calls[-1]
    assert state == {"claim": "x"}
    spec = questions["q"]
    assert spec["type"] == "choice"
    assert spec["instructions"] == "Pick one."
    assert spec["criteria"] == {"a": "first", "b": "second"}


def test_noul_mapping_with_descriptions():
    eng = _engine_with_fake({
        "noul": {"type": "noul", "noul": 0.7},
    })
    p = eng.noul("state", "Will it rain?", "likely rain", "likely dry")
    assert p == pytest.approx(0.7)
    _, questions = eng._decider.calls[-1]
    spec = questions["q"]
    assert spec["type"] == "noul"
    assert spec["criteria"] == {"true": "likely rain", "false": "likely dry"}


def test_noul_mapping_without_descriptions():
    eng = _engine_with_fake({
        "noul": {"type": "noul", "noul": 0.2},
    })
    assert eng.noul("state", "Will it rain?") == pytest.approx(0.2)
    _, questions = eng._decider.calls[-1]
    assert "criteria" not in questions["q"]


def test_score_uses_argmax_not_rounded_expectation():
    # expectation 1.4 would round to level "1"; the combined distribution's
    # argmax is level "2" — the adapter must report "2".
    eng = _engine_with_fake({
        "score": {"type": "score", "score": 1.4,
                  "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}},
    })
    level, probs, conf = eng.score(
        "state", "Rate it.", {"0": "low", "1": "mid", "2": "high"})
    assert level == "2"
    assert probs == {"0": 0.1, "1": 0.2, "2": 0.7}
    assert conf == pytest.approx(0.7)
    _, questions = eng._decider.calls[-1]
    spec = questions["q"]
    assert spec["type"] == "score"
    # legend keeps level order for the isolated-levels readout
    assert spec["criteria"] == ["low", "mid", "high"]


def test_score_none_description_becomes_empty_string():
    eng = _engine_with_fake({
        "score": {"type": "score", "score": 0.0,
                  "probabilities": {"0": 0.6, "1": 0.4}},
    })
    level, _, _ = eng.score("s", "i", {"0": None, "1": "high"})
    assert level == "0"
    _, questions = eng._decider.calls[-1]
    assert questions["q"]["criteria"] == ["", "high"]


def test_ask_without_load_raises():
    eng = DeciderEngine()
    with pytest.raises(RuntimeError, match="not loaded"):
        eng.choice("s", "i", {"a": None, "b": None})


def test_jeff1_engine_score_delegates_to_choice():
    eng = Jeff1Engine()
    seen = {}

    def fake_choice(state, instructions, criteria):
        seen["criteria"] = dict(criteria)
        return "1", {"0": 0.2, "1": 0.8}, 0.8

    eng.choice = fake_choice
    assert eng.score("s", "i", {"0": None, "1": None}) == \
        ("1", {"0": 0.2, "1": 0.8}, 0.8)
    assert seen["criteria"] == {"0": None, "1": None}


# -- end-to-end through the real handler -------------------------------------------


def _post(server, payload):
    url = f"http://127.0.0.1:{server.server_address[1]}/v1/jeff1/decide"
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


@pytest.fixture()
def decider_server():
    eng = _engine_with_fake({
        "choice": {"type": "choice", "choice": "b",
                   "probabilities": {"a": 0.25, "b": 0.75}},
        "noul": {"type": "noul", "noul": 0.8},
        "score": {"type": "score", "score": 1.1,
                  "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}},
    })
    server = ThreadingHTTPServer(("127.0.0.1", 0), Jeff1Handler)
    server.engine = eng
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    thread.join()
    server.server_close()


def test_decide_choice_end_to_end(decider_server):
    code, body = _post(decider_server, {
        "state": {"claim": "the sky is blue"},
        "instructions": "Pick the best label.",
        "criteria": {"a": "first", "b": "second"},
        "type": "choice",
    })
    assert code == 200
    assert body["type"] == "choice"
    assert body["label"] == "b"
    assert body["probabilities"] == {"a": 0.25, "b": 0.75}


def test_decide_noul_end_to_end(decider_server):
    code, body = _post(decider_server, {
        "state": "x", "instructions": "Will it succeed?", "type": "noul",
    })
    assert code == 200
    assert body["type"] == "noul"
    assert body["label"] == "yes"  # p_yes 0.8 >= 0.5
    assert body["probabilities"] == {"yes": 0.8, "no": 0.2}


def test_decide_score_end_to_end_uses_argmax(decider_server):
    code, body = _post(decider_server, {
        "state": "x", "instructions": "Rate severity.",
        "criteria": ["low", "medium", "high"], "type": "score",
    })
    assert code == 200
    assert body["type"] == "score"
    # argmax over probabilities -> "2"; the "score" expectation field (1.1,
    # which would round to "1") is ignored.
    assert body["level"] == "2"
    assert body["distribution"] == {"0": 0.1, "1": 0.2, "2": 0.7}


def test_healthz_reports_decider_model_id(decider_server):
    url = f"http://127.0.0.1:{decider_server.server_address[1]}/healthz"
    with urllib.request.urlopen(url, timeout=10) as resp:
        body = json.loads(resp.read())
    assert body["ok"] is True
    assert "decider-4b" in body["model"]
    assert body["loaded"] is True
