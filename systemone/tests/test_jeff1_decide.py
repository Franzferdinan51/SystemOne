"""Tests for POST /v1/jeff1/decide (systemone/jeff1_sidecar.py).

No model needed: the handler runs against a stub engine over real HTTP
sockets. Covers:

- the three decision types: response shape, TypeSafe confidence wiring
  (asserted against the repo helpers in systemone/api.py — confidence
  semantics adapted from Mapika/decider, Apache-2.0), criteria forwarding
- request validation -> 400 (bad type, missing/blank instructions, missing
  state, malformed criteria for each type, non-object body)
- lazy-load failure -> 503 (same pattern as the other endpoints)

TypeSafe confidence reference (systemone/api.py):
    choice: (n * p_max - 1) / (n - 1)
    score:  max(0, 1 - sum_i p_i * |i-k| / (n-1))
    noul:   max(P(yes), P(no))
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

from systemone.api import (  # noqa: E402
    choice_confidence,
    noul_confidence,
    score_confidence,
)
from systemone.jeff1_sidecar import Jeff1Handler  # noqa: E402


# -- stub engine ------------------------------------------------------------


class _StubEngine:
    """Jeff1Engine-shaped stub: canned choice/noul replies, no weights."""

    loaded = True
    load_error = None
    model_id = "stub"
    device = "cpu"

    def __init__(self, choice_reply=("a", {"a": 1.0}), noul_p=0.5):
        self.choice_reply = choice_reply
        self.noul_p = noul_p
        self.calls = []

    def load(self):
        pass

    def choice(self, state, instructions, criteria):
        self.calls.append(("choice", state, instructions, dict(criteria)))
        label, probs = self.choice_reply
        return label, dict(probs), max(probs.values())

    def noul(self, state, instructions, yes_desc=None, no_desc=None):
        self.calls.append(("noul", state, instructions, yes_desc, no_desc))
        return float(self.noul_p)

    def score(self, state, instructions, criteria):
        self.calls.append(("score", state, instructions, dict(criteria)))
        label, probs = self.choice_reply
        return label, dict(probs), max(probs.values())


class _FailingEngine(_StubEngine):
    """load() raises -> the handler must answer 503."""

    loaded = False

    def load(self):
        raise OSError("weights missing")


@pytest.fixture()
def sidecar():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Jeff1Handler)
    server.engine = _StubEngine()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    thread.join()
    server.server_close()


def _post(server, payload, raw=None):
    url = f"http://127.0.0.1:{server.server_address[1]}/v1/jeff1/decide"
    data = raw if raw is not None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


# -- choice -----------------------------------------------------------------


def test_choice_shape_and_typesafe_confidence(sidecar):
    sidecar.engine = _StubEngine(
        choice_reply=("a", {"a": 0.7, "b": 0.2, "c": 0.1}))
    code, body = _post(sidecar, {
        "state": {"claim": "the sky is blue"},
        "instructions": "Pick the best label.",
        "criteria": {"a": "first", "b": "second", "c": "third"},
        "type": "choice",
    })
    assert code == 200
    assert body["type"] == "choice"
    assert body["label"] == "a"
    # probabilities keep criteria order; confidence is the TypeSafe helper
    assert list(body["probabilities"]) == ["a", "b", "c"]
    assert body["probabilities"] == {"a": 0.7, "b": 0.2, "c": 0.1}
    assert body["confidence"] == round(choice_confidence([0.7, 0.2, 0.1]), 4)
    assert body["confidence"] == pytest.approx((3 * 0.7 - 1) / 2, abs=1e-3)
    assert isinstance(body["latency_ms"], (int, float))
    # the engine received the criteria and the request verbatim
    kind, state, instructions, criteria = sidecar.engine.calls[-1]
    assert kind == "choice"
    assert state == {"claim": "the sky is blue"}
    assert instructions == "Pick the best label."
    assert criteria == {"a": "first", "b": "second", "c": "third"}


def test_choice_uniform_confidence_is_zero(sidecar):
    third = 1.0 / 3.0
    sidecar.engine = _StubEngine(
        choice_reply=("a", {"a": third, "b": third, "c": third}))
    code, body = _post(sidecar, {
        "state": "x", "instructions": "pick",
        "criteria": {"a": "1", "b": "2", "c": "3"}, "type": "choice",
    })
    assert code == 200
    assert body["confidence"] == round(choice_confidence([third] * 3), 4) == 0.0


def test_choice_single_label_confidence_is_pmax(sidecar):
    sidecar.engine = _StubEngine(choice_reply=("only", {"only": 0.92}))
    code, body = _post(sidecar, {
        "state": "x", "instructions": "pick",
        "criteria": {"only": "the one"}, "type": "choice",
    })
    assert code == 200
    assert body["label"] == "only"
    assert body["confidence"] == round(choice_confidence([0.92]), 4) == 0.92


# -- noul -------------------------------------------------------------------


def test_noul_yes_no_criteria_absent(sidecar):
    sidecar.engine = _StubEngine(noul_p=0.8)
    code, body = _post(sidecar, {
        "state": "x", "instructions": "Is it true?", "type": "noul",
    })
    assert code == 200
    assert body["type"] == "noul"
    assert body["label"] == "yes"
    assert body["probabilities"] == {"yes": 0.8, "no": 0.2}
    assert body["confidence"] == round(noul_confidence(0.8), 4) == 0.8
    kind, _state, _instr, yes_desc, no_desc = sidecar.engine.calls[-1]
    assert kind == "noul"
    assert (yes_desc, no_desc) == (None, None)


def test_noul_no_with_dict_criteria(sidecar):
    sidecar.engine = _StubEngine(noul_p=0.3)
    code, body = _post(sidecar, {
        "state": "x", "instructions": "Is it true?", "type": "noul",
        "criteria": {"yes": "it holds", "no": "it fails"},
    })
    assert code == 200
    assert body["label"] == "no"
    assert body["probabilities"] == {"yes": 0.3, "no": 0.7}
    assert body["confidence"] == round(noul_confidence(0.3), 4) == 0.7
    kind, _state, _instr, yes_desc, no_desc = sidecar.engine.calls[-1]
    assert kind == "noul"
    assert (yes_desc, no_desc) == ("it holds", "it fails")


def test_noul_with_list_criteria(sidecar):
    sidecar.engine = _StubEngine(noul_p=0.55)
    code, body = _post(sidecar, {
        "state": "x", "instructions": "Is it true?", "type": "noul",
        "criteria": ["holds", "fails"],
    })
    assert code == 200
    assert body["label"] == "yes"
    assert body["confidence"] == round(noul_confidence(0.55), 4) == 0.55
    kind, _state, _instr, yes_desc, no_desc = sidecar.engine.calls[-1]
    assert (yes_desc, no_desc) == ("holds", "fails")


# -- score ------------------------------------------------------------------


def test_score_list_criteria(sidecar):
    sidecar.engine = _StubEngine(
        choice_reply=("2", {"0": 0.1, "1": 0.2, "2": 0.7}))
    code, body = _post(sidecar, {
        "state": "x", "instructions": "Rate severity.",
        "criteria": ["low", "medium", "high"], "type": "score",
    })
    assert code == 200
    assert body["type"] == "score"
    assert body["level"] == "2"
    assert body["distribution"] == {"0": 0.1, "1": 0.2, "2": 0.7}
    assert body["confidence"] == round(score_confidence([0.1, 0.2, 0.7]), 4)
    # engine.score runs over "0".."n-1" level labels
    kind, _state, _instr, criteria = sidecar.engine.calls[-1]
    assert kind == "score"
    assert criteria == {"0": "low", "1": "medium", "2": "high"}


def test_score_dict_criteria(sidecar):
    sidecar.engine = _StubEngine(
        choice_reply=("1", {"0": 0.2, "1": 0.6, "2": 0.2}))
    code, body = _post(sidecar, {
        "state": "x", "instructions": "Rate severity.",
        "criteria": {"0": "low", "1": "medium", "2": "high"}, "type": "score",
    })
    assert code == 200
    assert body["level"] == "1"
    assert body["confidence"] == round(score_confidence([0.2, 0.6, 0.2]), 4)


def test_score_delta_confidence_is_one(sidecar):
    sidecar.engine = _StubEngine(choice_reply=("0", {"0": 1.0, "1": 0.0}))
    code, body = _post(sidecar, {
        "state": "x", "instructions": "binary",
        "criteria": ["no", "yes"], "type": "score",
    })
    assert code == 200
    assert body["confidence"] == round(score_confidence([1.0, 0.0]), 4) == 1.0


# -- validation -> 400 --------------------------------------------------------


@pytest.mark.parametrize("payload", [
    {},
    {"state": "x", "instructions": "i"},                       # missing type
    {"state": "x", "instructions": "i", "type": "rank"},       # bad type
    {"state": "x", "type": "noul"},                             # no instructions
    {"state": "x", "instructions": "   ", "type": "noul"},     # blank instr
    {"instructions": "i", "type": "noul"},                      # no state
    {"state": "x", "instructions": "i", "type": "choice"},     # no criteria
    {"state": "x", "instructions": "i", "type": "choice",
     "criteria": {}},                                          # empty criteria
    {"state": "x", "instructions": "i", "type": "choice",
     "criteria": ["a", "b"]},                                  # wrong shape
    {"state": "x", "instructions": "i", "type": "choice",
     "criteria": {"a": 5}},                                    # bad description
    {"state": "x", "instructions": "i", "type": "score"},      # no criteria
    {"state": "x", "instructions": "i", "type": "score",
     "criteria": []},                                          # empty list
    {"state": "x", "instructions": "i", "type": "score",
     "criteria": {"0": "low", "2": "high"}},                   # gap in levels
    {"state": "x", "instructions": "i", "type": "score",
     "criteria": {"low": "0", "high": "2"}},                   # non-index keys
    {"state": "x", "instructions": "i", "type": "noul",
     "criteria": ["a", "b", "c"]},                             # bad noul shape
    {"state": "x", "instructions": "i", "type": "noul",
     "criteria": {"yes": 1, "no": "n"}},                       # bad yes desc
])
def test_decide_rejects_bad_requests(sidecar, payload):
    code, body = _post(sidecar, payload)
    assert code == 400
    assert "error" in body


def test_decide_rejects_non_object_body(sidecar):
    code, body = _post(sidecar, None, raw=b"[1, 2]")
    assert code == 400
    assert "error" in body


# -- lazy-load failure -> 503 -------------------------------------------------


def test_decide_503_when_weights_fail():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Jeff1Handler)
    server.engine = _FailingEngine()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        code, body = _post(server, {
            "state": "x", "instructions": "i", "type": "noul"})
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    assert code == 503
    assert "error" in body


# -- backend reporting --------------------------------------------------------


def test_decide_reports_active_backend(sidecar, monkeypatch):
    """The sidecar stamps decide replies with the active engine backend
    (SYSTEMONE_DECISION_BACKEND): jeff1 by default, decider when selected."""
    sidecar.engine = _StubEngine(noul_p=0.8)
    payload = {"state": "x", "instructions": "Is it true?", "type": "noul"}

    monkeypatch.setenv("SYSTEMONE_DECISION_BACKEND", "decider")
    code, body = _post(sidecar, payload)
    assert code == 200
    assert body["backend"] == "decider"

    monkeypatch.delenv("SYSTEMONE_DECISION_BACKEND", raising=False)
    code, body = _post(sidecar, payload)
    assert code == 200
    assert body["backend"] == "jeff1"
