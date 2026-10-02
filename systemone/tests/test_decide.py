"""Tests for POST /v1/systemone/decide (systemone/shim.py).

No model needed: HTTP tests run against a stub engine, and Jeff-1 is
either monkeypatched or served by a fake loopback sidecar. Covers:

- request validation -> 400 (bad type, missing state/instructions,
  malformed criteria), mirroring the sidecar's decide schema
- primary path: proxy to the Jeff-1 sidecar; the reply comes back with
  backend "decider" and the forwarded request is the decide schema
- fail-open fallback: sidecar down / 404 / malformed -> the local
  GLiClass machinery answers with backend "fallback", sidecar-shaped
  response, TypeSafe confidence wiring
- per-type temperature info appears only when the engine carries a
  fitted per-answer-type calibrator
- metric recording: decide_records grows and decide_metrics_summary()
  computes ECE/Brier/NLL via systemone.metrics over gold-annotated rows
- decide_via_jeff1() fail-open behavior against a real loopback server
"""

import json
import os
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from systemone import shim  # noqa: E402
from systemone.calibration import PerTypeTemperatureCalibrator  # noqa: E402
from systemone.jeff1 import decide_via_jeff1  # noqa: E402
from systemone.shim import decide_metrics_summary, serve  # noqa: E402


class StubEngine:
    """systemone()-shaped stub: canned answers per question type."""

    model_name = "stub"
    systemone_calls = None

    def __init__(self):
        self.systemone_calls = []

    def systemone(self, state, questions, batch_size=32):
        self.systemone_calls.append((state, questions))
        answers = {}
        for q in questions:
            qtype = q["type"]
            if qtype == "choice":
                labs = list(q["options"])
                n = len(labs)
                probs = {lab: (0.7 if i == 0 else 0.3 / (n - 1) if n > 1 else 1.0)
                         for i, lab in enumerate(labs)}
                answers[q["name"]] = {
                    "type": "choice", "choice": labs[0],
                    "probabilities": probs, "confidence": 0.4,
                }
            elif qtype == "score":
                labs = list(q["levels"])
                n = len(labs)
                probs = {lab: (0.6 if i == 0 else 0.4 / (n - 1) if n > 1 else 1.0)
                         for i, lab in enumerate(labs)}
                answers[q["name"]] = {
                    "type": "score", "level": labs[0],
                    "distribution": probs, "confidence": 0.5,
                }
            else:  # noul
                answers[q["name"]] = {
                    "type": "noul", "probability": 0.75,
                    "answer": True, "confidence": 0.75,
                }
        answers["_meta"] = {"model": "stub", "latency_ms": 1.0}
        return answers


@pytest.fixture()
def server(monkeypatch):
    monkeypatch.setenv("SYSTEMONE_LOG_DISABLE", "1")
    engine = StubEngine()
    srv = serve(0, engine=engine)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv, port, engine
    srv.shutdown()
    t.join(timeout=5)
    srv.server_close()


def _post_decide(port, payload, raw=None):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/systemone/decide",
        data=raw if raw is not None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.request.HTTPError as e:
        return e.code, json.loads(e.read())


def _fallback(monkeypatch, engine, reply=None):
    """Force the fail-open path; returns the list of forwarded payloads."""
    seen = []

    def fake_decide(payload, timeout=None):
        seen.append(payload)
        return reply

    monkeypatch.setattr(shim, "decide_via_jeff1", fake_decide)
    engine.systemone_calls.clear()
    return seen


# -- validation -> 400 -------------------------------------------------------


@pytest.mark.parametrize("payload", [
    {},
    {"state": "x", "instructions": "i"},                       # missing type
    {"state": "x", "instructions": "i", "type": "rank"},        # bad type
    {"state": "x", "type": "noul"},                             # no instructions
    {"state": "x", "instructions": "   ", "type": "noul"},      # blank instr
    {"instructions": "i", "type": "noul"},                      # no state
    {"state": "x", "instructions": "i", "type": "choice"},      # no criteria
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
    {"state": "x", "instructions": "i", "type": "noul",
     "criteria": {"maybe": "m"}},                             # unknown keys
])
def test_decide_rejects_bad_requests(server, monkeypatch, payload):
    _, port, engine = server
    seen = _fallback(monkeypatch, engine)
    code, body = _post_decide(port, payload)
    assert code == 400
    assert "error" in body
    assert seen == []  # validation fails before any Jeff-1 attempt


def test_decide_accepts_true_false_noul_criteria(server, monkeypatch):
    _, port, engine = server
    _fallback(monkeypatch, engine)
    code, body = _post_decide(port, {
        "state": "x", "instructions": "Is it up?", "type": "noul",
        "criteria": {"true": "Service is up.", "false": "Service is down."},
    })
    assert code == 200
    assert body["probabilities"] == {"yes": 0.75, "no": 0.25}
    statement = engine.systemone_calls[0][1][0]["statement"]
    assert "Service is up." in statement and "Service is down." in statement


def test_decide_rejects_non_object_body(server, monkeypatch):
    _, port, engine = server
    _fallback(monkeypatch, engine)
    code, body = _post_decide(port, None, raw=b"[1, 2]")
    assert code == 400
    assert "error" in body


# -- primary path: proxy to Jeff-1 -------------------------------------------


def test_proxy_path_backend_decider(server, monkeypatch):
    _, port, engine = server
    forwarded = []

    def fake_decide(payload, timeout=None):
        forwarded.append(payload)
        return {"type": "choice", "label": "b",
                "probabilities": {"a": 0.2, "b": 0.8},
                "confidence": 0.6}

    monkeypatch.setattr(shim, "decide_via_jeff1", fake_decide)
    code, body = _post_decide(port, {
        "state": "some page text",
        "instructions": "Pick the best.",
        "criteria": {"a": "first", "b": "second"},
        "type": "choice",
    })
    assert code == 200
    assert body["backend"] == "decider"
    assert body["type"] == "choice"
    assert body["label"] == "b"
    assert body["probabilities"] == {"a": 0.2, "b": 0.8}
    assert body["confidence"] == 0.6
    assert isinstance(body["latency_ms"], (int, float))
    assert "temperature" not in body
    # forwarded verbatim in the sidecar's decide schema
    assert forwarded == [{
        "state": "some page text",
        "instructions": "Pick the best.",
        "criteria": {"a": "first", "b": "second"},
        "type": "choice",
    }]
    # the local engine was never touched on the primary path
    assert engine.systemone_calls == []


def test_proxy_path_noul_passthrough(server, monkeypatch):
    _, port, engine = server
    monkeypatch.setattr(
        shim, "decide_via_jeff1",
        lambda payload, timeout=None: {
            "type": "noul", "label": "no",
            "probabilities": {"yes": 0.3, "no": 0.7}, "confidence": 0.7})
    code, body = _post_decide(port, {
        "state": "x", "instructions": "Is it true?", "type": "noul",
        "criteria": {"yes": "holds", "no": "fails"},
    })
    assert code == 200
    assert body["backend"] == "decider"
    assert body["label"] == "no"
    assert body["probabilities"] == {"yes": 0.3, "no": 0.7}


# -- fail-open fallback ------------------------------------------------------


def test_fallback_choice_shape(server, monkeypatch):
    _, port, engine = server
    _fallback(monkeypatch, engine)
    code, body = _post_decide(port, {
        "state": "some page text",
        "instructions": "Pick the best.",
        "criteria": {"a": "first", "b": "second"},
        "type": "choice",
    })
    assert code == 200
    assert body["backend"] == "fallback"
    assert body["type"] == "choice"
    assert body["label"] == "a"
    assert list(body["probabilities"]) == ["a", "b"]  # criteria order kept
    assert body["probabilities"]["a"] == pytest.approx(0.7)
    assert body["confidence"] == pytest.approx(0.4)
    assert body["model"] == "stub"
    assert isinstance(body["latency_ms"], (int, float))
    assert "temperature" not in body  # stub engine carries no calibrator
    # the fallback went through the legacy decision machinery
    assert len(engine.systemone_calls) == 1
    state, questions = engine.systemone_calls[0]
    assert state == "some page text"
    assert questions[0]["type"] == "choice"
    assert questions[0]["options"] == ["a", "b"]
    assert "Pick the best." in questions[0]["prompt"]
    assert "first" in questions[0]["prompt"]  # criterion descriptions kept


def test_fallback_noul_shape(server, monkeypatch):
    _, port, engine = server
    _fallback(monkeypatch, engine)
    code, body = _post_decide(port, {
        "state": "x", "instructions": "Is it true?", "type": "noul",
        "criteria": {"yes": "it holds", "no": "it fails"},
    })
    assert code == 200
    assert body["backend"] == "fallback"
    assert body["type"] == "noul"
    assert body["label"] == "yes"
    assert body["probabilities"] == {"yes": 0.75, "no": 0.25}
    assert body["confidence"] == pytest.approx(0.75)


def test_fallback_score_shape(server, monkeypatch):
    _, port, engine = server
    _fallback(monkeypatch, engine)
    code, body = _post_decide(port, {
        "state": "x", "instructions": "Rate severity.",
        "criteria": ["low", "high"], "type": "score",
    })
    assert code == 200
    assert body["backend"] == "fallback"
    assert body["type"] == "score"
    assert body["level"] == "0"
    assert list(body["distribution"]) == ["0", "1"]
    assert body["distribution"]["0"] == pytest.approx(0.6)


def test_fallback_temperature_info_when_calibrated(server, monkeypatch):
    srv, port, engine = server
    _fallback(monkeypatch, engine)
    cal = PerTypeTemperatureCalibrator(min_rows=2)
    cal.fit([
        {"type": "choice", "gold": 0, "probs": [0.7, 0.3]},
        {"type": "choice", "gold": 1, "probs": [0.4, 0.6]},
        {"type": "noul", "gold": 0, "probs": [0.8, 0.2]},
        {"type": "noul", "gold": 1, "probs": [0.3, 0.7]},
    ])
    engine.calibrator = cal
    try:
        code, body = _post_decide(port, {
            "state": "x", "instructions": "Pick.",
            "criteria": {"a": "1", "b": "2"}, "type": "choice",
        })
    finally:
        del engine.calibrator
    assert code == 200
    assert body["backend"] == "fallback"
    temp = body["temperature"]
    assert temp["applied"] == pytest.approx(cal.temperature_for("choice"))
    assert temp["pooled"] == pytest.approx(cal.temperature_)
    assert set(temp["per_type"]) == {"choice", "noul"}


def test_fallback_records_metric(server, monkeypatch):
    srv, port, engine = server
    _fallback(monkeypatch, engine)
    before = len(srv.decide_records)
    _post_decide(port, {
        "state": "x", "instructions": "Pick.",
        "criteria": {"a": "1", "b": "2"}, "type": "choice",
    })
    assert len(srv.decide_records) == before + 1
    rec = srv.decide_records[-1]
    assert rec["type"] == "choice"
    assert rec["labels"] == ["a", "b"]
    assert rec["probs"] == pytest.approx([0.7, 0.3])
    assert rec["backend"] == "fallback"
    assert "ts" in rec


def test_decide_metrics_summary():
    recs = [
        {"type": "choice", "probs": [0.7, 0.2, 0.1], "gold": 0},
        {"type": "choice", "probs": [0.2, 0.7, 0.1], "gold": 1},
        {"type": "noul", "probs": [0.8, 0.2], "gold": 0},
        {"type": "score", "probs": [0.1, 0.7, 0.2]},          # no gold: skipped
        {"type": "bogus", "probs": [0.5, 0.5], "gold": 0},    # bad type: skipped
        {"type": "choice", "probs": [0.5, 0.5], "gold": 9},   # bad gold: skipped
    ]
    out = decide_metrics_summary(recs)
    assert set(out) == {"choice", "noul"}
    for row in out.values():
        assert {"ece_15", "brier", "nll"} <= set(row)


def test_jeff1_disabled_forces_fallback(server, monkeypatch):
    # SYSTEMONE_JEFF1=0 makes decide_via_jeff1 fail open without touching HTTP
    monkeypatch.setenv("SYSTEMONE_JEFF1", "0")
    _, port, engine = server
    code, body = _post_decide(port, {
        "state": "x", "instructions": "Pick.",
        "criteria": {"a": "1"}, "type": "choice",
    })
    assert code == 200
    assert body["backend"] == "fallback"
    assert body["label"] == "a"


# -- decide_via_jeff1 against a real loopback server --------------------------


class _FakeSidecar(BaseHTTPRequestHandler):
    mode = "ok"
    seen = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        _FakeSidecar.seen.append(json.loads(self.rfile.read(length) or b"{}"))
        mode = _FakeSidecar.mode
        if self.path != "/v1/jeff1/decide":
            body, code = {"error": "not found"}, 404
        elif mode == "ok":
            body, code = ({"type": "choice", "label": "a",
                           "probabilities": {"a": 1.0}, "confidence": 1.0}, 200)
        elif mode == "garbage":
            body, code = ([1, 2, 3], 200)
        elif mode == "wrongtype":
            body, code = ({"type": "rank"}, 200)
        else:
            body, code = ({"error": "nope"}, 404)
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        pass


@pytest.fixture()
def fake_sidecar(monkeypatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FakeSidecar)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    monkeypatch.setenv("SYSTEMONE_JEFF1", "1")
    monkeypatch.setenv("SYSTEMONE_JEFF1_URL", f"http://127.0.0.1:{port}")
    _FakeSidecar.mode = "ok"
    _FakeSidecar.seen = []
    yield port
    srv.shutdown()
    t.join(timeout=5)
    srv.server_close()


def test_decide_via_jeff1_ok(fake_sidecar):
    payload = {"state": "x", "instructions": "i",
               "criteria": {"a": "1"}, "type": "choice"}
    out = decide_via_jeff1(payload, timeout=5)
    assert out["type"] == "choice"
    assert out["label"] == "a"
    assert _FakeSidecar.seen[-1] == payload


def test_decide_via_jeff1_404_is_fail_open(fake_sidecar):
    _FakeSidecar.mode = "notfound"
    assert decide_via_jeff1({"type": "choice"}, timeout=5) is None


def test_decide_via_jeff1_malformed_body_is_fail_open(fake_sidecar):
    _FakeSidecar.mode = "garbage"
    assert decide_via_jeff1({"type": "choice"}, timeout=5) is None
    _FakeSidecar.mode = "wrongtype"
    assert decide_via_jeff1({"type": "choice"}, timeout=5) is None


def test_decide_via_jeff1_refused_is_fail_open(monkeypatch):
    # bind-then-close: a port nothing listens on
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FakeSidecar)
    port = srv.server_address[1]
    srv.server_close()
    monkeypatch.setenv("SYSTEMONE_JEFF1", "1")
    monkeypatch.setenv("SYSTEMONE_JEFF1_URL", f"http://127.0.0.1:{port}")
    assert decide_via_jeff1({"type": "choice"}, timeout=2) is None


def test_decide_via_jeff1_disabled_is_fail_open(monkeypatch, fake_sidecar):
    monkeypatch.setenv("SYSTEMONE_JEFF1", "0")
    assert decide_via_jeff1({"type": "choice"}, timeout=5) is None
    assert _FakeSidecar.seen == []  # never even attempted


def test_proxy_path_preserves_sidecar_backend(server, monkeypatch):
    """The shim passes through the backend the sidecar reported (decider),
    instead of stamping every proxied answer with a fixed label."""
    _, port, engine = server
    monkeypatch.setattr(
        shim, "decide_via_jeff1",
        lambda payload, timeout=None: {
            "type": "noul", "label": "yes",
            "probabilities": {"yes": 0.9, "no": 0.1}, "confidence": 0.9,
            "backend": "decider"})
    code, body = _post_decide(port, {
        "state": "x", "instructions": "Is it true?", "type": "noul",
    })
    assert code == 200
    assert body["backend"] == "decider"
    assert body["label"] == "yes"
