"""Tests for the Jeff-1 second decision head (systemone/jeff1.py + shim wiring).

No model needed: the shim-side client talks to stub HTTP sidecars, blending
is pure, and fail-open paths (refused connection, timeout, malformed reply)
are exercised with real sockets. Covers:

- config parsing: default-on, SYSTEMONE_JEFF1=0 disables, URL/timeout/blend
  defaults and overrides
- rank_plans_via_jeff1 / second_opinion: parsed reply, malformed -> None,
  connection refused -> None (no raise), timeout respected
- blend_rankings: 50/50 blend, cost penalty re-applied, re-sorted, fail-open
  when a plan is missing from the Jeff-1 reply
- shim wiring: uncertain route records jeff1_second_opinion (advisory note
  appended on disagreement); certain routes NEVER touch Jeff-1;
  SYSTEMONE_JEFF1=0 skips it entirely
- POST /v1/systemone/rank-plans: stub sidecar -> jeff1.consulted=true and
  the ranking flips under the blend; sidecar down -> consulted=false,
  GLiClass-only ranking, still 200
"""

import json
import os
import socket
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from systemone import jeff1, shim
from systemone.jeff1 import (
    blend_rankings,
    jeff1_blend_weight,
    jeff1_enabled,
    jeff1_timeout,
    jeff1_url,
    rank_plans_via_jeff1,
    second_opinion,
)
from systemone.shim import serve


# -- stub sidecar --------------------------------------------------------

class _StubSidecarHandler(BaseHTTPRequestHandler):
    """Canned Jeff-1 sidecar. Responses set on server.responses {path: body}."""
    server_version = "StubJeff1/0"

    def _send(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        server = self.server
        with server.lock:
            server.hits.append(self.path)
            delay = server.delays.get(self.path, 0)
            body = server.responses.get(self.path)
        if delay:
            time.sleep(delay)
        if body is None:
            self._send(404, {"error": "not found"})
        else:
            self._send(200, body)

    def log_message(self, fmt, *args):
        pass


def _stub_sidecar(responses=None, delays=None):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubSidecarHandler)
    server.responses = responses or {}
    server.delays = delays or {}
    server.hits = []
    server.lock = threading.Lock()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# -- stub GLiClass engine (for the shim roundtrip tests) -------------------

class _StubEngine:
    """Deterministic engine: choice weights the first option; noul answers
    P(yes) from the prob map keyed by question name."""

    model_name = "stub"

    def __init__(self, noul_probs=None):
        self.noul_probs = noul_probs or {}

    def systemone(self, state, questions):
        out = {}
        for q in questions:
            name = q["name"]
            if q["type"] == "choice":
                opts = q["options"]
                rest = (1.0 - 0.7) / (len(opts) - 1) if len(opts) > 1 else 0.0
                probs = {o: (0.7 if i == 0 else rest)
                         for i, o in enumerate(opts)}
                out[name] = {"type": "choice", "choice": opts[0],
                             "probabilities": probs, "confidence": 0.7}
            else:  # noul
                p = self.noul_probs.get(name, 0.5)
                out[name] = {"type": "noul", "probability": p,
                             "answer": p >= 0.5,
                             "confidence": abs(p - 0.5) * 2}
        out["_meta"] = {"latency_ms": 1.0}
        return out


def _registry():
    def tier(desc):
        return {"model_id": "stub-model", "description": desc,
                "models": [{"model_id": "stub-model", "cost": 1.0}]}
    return {"tiers": {"economy": tier("cheap tier"),
                      "balanced": tier("mid tier"),
                      "heavy": tier("strong tier")}}


def _serve_shim(engine):
    server = serve(port=0, engine=engine, registry=_registry())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _post(port, path, payload, timeout=10):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode())


# -- config ---------------------------------------------------------------

def test_jeff1_default_on(monkeypatch):
    for var in ("SYSTEMONE_JEFF1", "SYSTEMONE_JEFF1_URL",
                "SYSTEMONE_JEFF1_TIMEOUT", "SYSTEMONE_JEFF1_BLEND"):
        monkeypatch.delenv(var, raising=False)
    assert jeff1_enabled() is True
    assert jeff1_url() == "http://127.0.0.1:8079"
    assert jeff1_timeout() == 2.5
    assert jeff1_blend_weight() == 0.5


@pytest.mark.parametrize("val", ["0", "false", "no", "off", "disabled", " 0 "])
def test_jeff1_env_disables(monkeypatch, val):
    monkeypatch.setenv("SYSTEMONE_JEFF1", val)
    assert jeff1_enabled() is False


@pytest.mark.parametrize("val", ["1", "true", "yes", "on", "banana", ""])
def test_jeff1_non_off_values_stay_on(monkeypatch, val):
    monkeypatch.setenv("SYSTEMONE_JEFF1", val)
    assert jeff1_enabled() is True


def test_jeff1_url_and_timeout_overrides(monkeypatch):
    monkeypatch.setenv("SYSTEMONE_JEFF1_URL", "http://10.0.0.9:8079")
    monkeypatch.setenv("SYSTEMONE_JEFF1_TIMEOUT", "7.5")
    monkeypatch.setenv("SYSTEMONE_JEFF1_BLEND", "0.8")
    assert jeff1_url() == "http://10.0.0.9:8079"
    assert jeff1_timeout() == 7.5
    assert jeff1_blend_weight() == 0.8


@pytest.mark.parametrize("val", ["", "abc", "-3", "0"])
def test_jeff1_bad_timeout_falls_back_to_default(monkeypatch, val):
    monkeypatch.setenv("SYSTEMONE_JEFF1_TIMEOUT", val)
    assert jeff1_timeout() == 2.5


# -- fail-open client behavior --------------------------------------------

def test_rank_plans_connection_refused_returns_none(monkeypatch):
    monkeypatch.setenv("SYSTEMONE_JEFF1_URL",
                       f"http://127.0.0.1:{_free_port()}")
    t0 = time.perf_counter()
    assert rank_plans_via_jeff1("task", [{"id": "a", "text": "t"}]) is None
    assert time.perf_counter() - t0 < 2.0  # fast refusal, no hang


def test_rank_plans_disabled_never_touches_network(monkeypatch):
    monkeypatch.setenv("SYSTEMONE_JEFF1", "0")
    monkeypatch.setenv("SYSTEMONE_JEFF1_URL",
                       f"http://127.0.0.1:{_free_port()}")
    assert rank_plans_via_jeff1("task", [{"id": "a", "text": "t"}]) is None
    assert second_opinion("task", {"tier": "balanced"}) is None


def test_timeout_respected(monkeypatch):
    server = _stub_sidecar(
        responses={"/v1/jeff1/rank-plans": {"ranking": []}},
        delays={"/v1/jeff1/rank-plans": 5.0})
    port = server.server_address[1]
    monkeypatch.setenv("SYSTEMONE_JEFF1_URL", f"http://127.0.0.1:{port}")
    try:
        t0 = time.perf_counter()
        assert rank_plans_via_jeff1("task", [{"id": "a", "text": "t"}],
                                    timeout=0.4) is None
        assert time.perf_counter() - t0 < 3.0
    finally:
        server.shutdown()


def test_rank_plans_parses_reply(monkeypatch):
    server = _stub_sidecar(responses={
        "/v1/jeff1/rank-plans": {
            "ranking": [{"id": "a", "p_success": 0.9},
                        {"id": "b", "p_success": 0.12345}]}})
    port = server.server_address[1]
    monkeypatch.setenv("SYSTEMONE_JEFF1_URL", f"http://127.0.0.1:{port}")
    try:
        out = rank_plans_via_jeff1("task",
                                   [{"id": "a", "text": "x"},
                                    {"id": "b", "text": "y"}])
        assert out == [{"id": "a", "p_success": 0.9},
                       {"id": "b", "p_success": 0.12345}]
    finally:
        server.shutdown()


@pytest.mark.parametrize("bad", [
    {"ranking": "nope"},
    {"ranking": []},
    {"ranking": [{"id": "a"}]},                 # missing p_success
    {"ranking": [{"id": "a", "p_success": "x"}]},
    {"nope": 1},
])
def test_rank_plans_malformed_reply_returns_none(monkeypatch, bad):
    server = _stub_sidecar(
        responses={"/v1/jeff1/rank-plans": bad})
    port = server.server_address[1]
    monkeypatch.setenv("SYSTEMONE_JEFF1_URL", f"http://127.0.0.1:{port}")
    try:
        assert rank_plans_via_jeff1("task", [{"id": "a", "text": "x"}]) is None
    finally:
        server.shutdown()


def test_second_opinion_parses_reply(monkeypatch):
    server = _stub_sidecar(responses={
        "/v1/jeff1/second-opinion": {
            "tier": "heavy", "confidence": 0.71, "agree": False,
            "rationale": "heavy fits better"}})
    port = server.server_address[1]
    monkeypatch.setenv("SYSTEMONE_JEFF1_URL", f"http://127.0.0.1:{port}")
    try:
        out = second_opinion("task", {"tier": "balanced",
                                      "candidates": [{"tier": "heavy",
                                                      "description": "strong"}]})
        assert out == {"tier": "heavy", "confidence": 0.71, "agree": False,
                       "rationale": "heavy fits better"}
    finally:
        server.shutdown()


def test_second_opinion_malformed_or_down_returns_none(monkeypatch):
    monkeypatch.setenv("SYSTEMONE_JEFF1_URL",
                       f"http://127.0.0.1:{_free_port()}")
    assert second_opinion("task", {"tier": "balanced"}) is None


# -- blending ---------------------------------------------------------------

def _g(pid, p, steps=1):
    penalty = round(steps * 1.0 / 100.0, 4)
    return {"id": pid, "score": round(p - penalty, 4), "p_success": p,
            "cost_penalty": penalty, "est_steps": steps}


def test_blend_rankings_5050_and_resort():
    gliclass = [_g("a", 0.9), _g("b", 0.4)]
    jeff = [{"id": "a", "p_success": 0.01}, {"id": "b", "p_success": 0.99}]
    out = blend_rankings(gliclass, jeff, tier_cost=1.0)
    by_id = {r["id"]: r for r in out}
    assert by_id["a"]["p_success"] == round(0.5 * 0.9 + 0.5 * 0.01, 4)
    assert by_id["b"]["p_success"] == round(0.5 * 0.4 + 0.5 * 0.99, 4)
    assert by_id["a"]["jeff1_blended"] is True
    # re-sorted: b wins the blend even though GLiClass had a first
    assert [r["id"] for r in out] == ["b", "a"]
    # cost penalty re-applied after the blend
    assert by_id["b"]["cost_penalty"] == 0.01
    assert by_id["b"]["score"] == round(by_id["b"]["p_success"] - 0.01, 4)


def test_blend_missing_jeff_plan_keeps_gliclass_score():
    gliclass = [_g("a", 0.9), _g("b", 0.4)]
    jeff = [{"id": "a", "p_success": 0.8}]  # no "b"
    out = blend_rankings(gliclass, jeff, tier_cost=None)
    by_id = {r["id"]: r for r in out}
    assert by_id["b"]["p_success"] == 0.4
    assert by_id["b"]["jeff1_blended"] is False
    assert by_id["b"]["cost_penalty"] == 0.0


def test_blend_weight_env_override(monkeypatch):
    monkeypatch.setenv("SYSTEMONE_JEFF1_BLEND", "1.0")
    out = blend_rankings([_g("a", 0.2)], [{"id": "a", "p_success": 0.9}],
                         tier_cost=None)
    assert out[0]["p_success"] == 0.9


# -- shim wiring: uncertain routes consult Jeff-1 ----------------------------

def _scoring_ctx_uncertain():
    return {
        "calibration": {"temperature": 1.0},
        "tools": [],
        "registry": {"tiers": {"economy": {"description": "cheap"},
                               "balanced": {"description": "mid"},
                               "heavy": {"description": "strong"}}},
    }


def _route(tier="balanced"):
    return {"model_id": "m", "tier": tier, "rationale": "test route",
            "confidence": 0.5, "probabilities": {}, "effort": "medium"}


def test_uncertain_route_records_second_opinion(monkeypatch):
    from systemone.scoring import reset_tuning
    reset_tuning()
    monkeypatch.setenv("SYSTEMONE_MARGIN_FLOOR", "0.99")  # everything uncertain
    server = _stub_sidecar(responses={
        "/v1/jeff1/second-opinion": {
            "tier": "heavy", "confidence": 0.71, "agree": False,
            "rationale": "heavy fits better"}})
    port = server.server_address[1]
    monkeypatch.setenv("SYSTEMONE_JEFF1", "1")
    monkeypatch.setenv("SYSTEMONE_JEFF1_URL", f"http://127.0.0.1:{port}")
    try:
        route = _route()
        shim._apply_scoring(_StubEngine(), "some ambiguous task", route,
                            {"economy": 0.45, "balanced": 0.40, "heavy": 0.15},
                            _scoring_ctx_uncertain())
        assert route["uncertain"] is True
        opinion = route["jeff1_second_opinion"]
        assert opinion == {"tier": "heavy", "confidence": 0.71,
                           "agree": False, "rationale": "heavy fits better"}
        # disagreement noted on the rationale, tier untouched
        assert "advisory" in route["rationale"]
        assert route["tier"] == "balanced"
    finally:
        server.shutdown()
        monkeypatch.delenv("SYSTEMONE_MARGIN_FLOOR", raising=False)


def test_uncertain_route_agreeing_opinion_no_rationale_note(monkeypatch):
    from systemone.scoring import reset_tuning
    reset_tuning()
    monkeypatch.setenv("SYSTEMONE_MARGIN_FLOOR", "0.99")
    server = _stub_sidecar(responses={
        "/v1/jeff1/second-opinion": {
            "tier": "balanced", "confidence": 0.8, "agree": True,
            "rationale": "balanced is fine"}})
    port = server.server_address[1]
    monkeypatch.setenv("SYSTEMONE_JEFF1", "1")
    monkeypatch.setenv("SYSTEMONE_JEFF1_URL", f"http://127.0.0.1:{port}")
    try:
        route = _route()
        shim._apply_scoring(_StubEngine(), "some ambiguous task", route,
                            {"economy": 0.45, "balanced": 0.40, "heavy": 0.15},
                            _scoring_ctx_uncertain())
        assert route["jeff1_second_opinion"]["agree"] is True
        # no disagreement note when the heads agree (the effort-bump note
        # from the uncertain path itself is still there — that's not Jeff-1)
        assert "second opinion" not in route["rationale"]
    finally:
        server.shutdown()
        monkeypatch.delenv("SYSTEMONE_MARGIN_FLOOR", raising=False)


def test_certain_route_never_calls_jeff1(monkeypatch):
    from systemone.scoring import reset_tuning
    reset_tuning()
    monkeypatch.setenv("SYSTEMONE_MARGIN_FLOOR", "0.01")  # nothing uncertain
    calls = []

    def boom(*a, **k):
        calls.append((a, k))
        raise AssertionError("jeff1 must not be called on certain routes")

    monkeypatch.setattr(shim, "jeff1_second_opinion", boom)
    monkeypatch.setattr(jeff1, "_post", boom)
    route = _route()
    shim._apply_scoring(_StubEngine(), "what is 2+2?", route,
                        {"economy": 0.90, "balanced": 0.08, "heavy": 0.02},
                        _scoring_ctx_uncertain())
    assert route["uncertain"] is False
    assert calls == []
    assert "jeff1_second_opinion" not in route
    monkeypatch.delenv("SYSTEMONE_MARGIN_FLOOR", raising=False)


def test_jeff1_disabled_skips_second_opinion_on_uncertain(monkeypatch):
    from systemone.scoring import reset_tuning
    reset_tuning()
    monkeypatch.setenv("SYSTEMONE_MARGIN_FLOOR", "0.99")
    monkeypatch.setenv("SYSTEMONE_JEFF1", "0")
    monkeypatch.setenv("SYSTEMONE_JEFF1_URL",
                       f"http://127.0.0.1:{_free_port()}")

    def boom(*a, **k):
        raise AssertionError("disabled head must not issue HTTP")

    monkeypatch.setattr(jeff1, "_post", boom)
    route = _route()
    shim._apply_scoring(_StubEngine(), "some ambiguous task", route,
                        {"economy": 0.45, "balanced": 0.40, "heavy": 0.15},
                        _scoring_ctx_uncertain())
    assert route["uncertain"] is True
    assert "jeff1_second_opinion" not in route
    monkeypatch.delenv("SYSTEMONE_MARGIN_FLOOR", raising=False)


# -- shim wiring: rank-plans blend --------------------------------------------

def test_rank_plans_http_blends_with_sidecar(monkeypatch):
    sidecar = _stub_sidecar(responses={
        "/v1/jeff1/rank-plans": {
            "ranking": [{"id": "b", "p_success": 0.99},
                        {"id": "a", "p_success": 0.01}]},
        "/v1/jeff1/second-opinion": {
            "tier": "balanced", "confidence": 0.8, "agree": True,
            "rationale": "fine"}})
    sport = sidecar.server_address[1]
    monkeypatch.setenv("SYSTEMONE_JEFF1", "1")
    monkeypatch.setenv("SYSTEMONE_JEFF1_URL", f"http://127.0.0.1:{sport}")
    engine = _StubEngine(noul_probs={"plan_0": 0.9, "plan_1": 0.4})
    server = _serve_shim(engine)
    port = server.server_address[1]
    try:
        status, body = _post(port, "/v1/systemone/rank-plans", {
            "task": "migrate the database",
            "plans": [{"id": "a", "text": "1. backup\n2. migrate"},
                      {"id": "b", "text": "1. migrate"}],
        })
        assert status == 200
        info = body["jeff1"]
        assert info["consulted"] is True
        assert info["blended"] is True
        assert info["latency_ms"] is not None
        # GLiClass had a first (0.9); the 50/50 blend flips it to b
        assert [r["id"] for r in body["ranking"]] == ["b", "a"]
        assert all(r["jeff1_blended"] for r in body["ranking"])
    finally:
        server.shutdown()
        sidecar.shutdown()


def test_rank_plans_http_sidecar_down_fails_open(monkeypatch):
    monkeypatch.setenv("SYSTEMONE_JEFF1", "1")
    monkeypatch.setenv("SYSTEMONE_JEFF1_URL",
                       f"http://127.0.0.1:{_free_port()}")
    engine = _StubEngine(noul_probs={"plan_0": 0.9, "plan_1": 0.4})
    server = _serve_shim(engine)
    port = server.server_address[1]
    try:
        status, body = _post(port, "/v1/systemone/rank-plans", {
            "task": "migrate the database",
            "plans": [{"id": "a", "text": "1. backup\n2. migrate"},
                      {"id": "b", "text": "1. migrate"}],
        })
        assert status == 200
        assert body["jeff1"]["consulted"] is False
        assert body["jeff1"]["blended"] is False
        # GLiClass-only ranking stands
        assert [r["id"] for r in body["ranking"]] == ["a", "b"]
    finally:
        server.shutdown()


def test_rank_plans_http_jeff1_disabled_no_attempt(monkeypatch):
    monkeypatch.setenv("SYSTEMONE_JEFF1", "0")
    engine = _StubEngine(noul_probs={"plan_0": 0.9, "plan_1": 0.4})
    server = _serve_shim(engine)
    port = server.server_address[1]
    try:
        status, body = _post(port, "/v1/systemone/rank-plans", {
            "task": "migrate the database",
            "plans": [{"id": "a", "text": "x"}, {"id": "b", "text": "y"}],
        })
        assert status == 200
        assert body["jeff1"]["consulted"] is False
    finally:
        server.shutdown()
