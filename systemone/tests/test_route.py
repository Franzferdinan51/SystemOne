"""Tests for POST /v1/systemone/route and shim latency logging.

No model needed — routing logic is pure and HTTP tests use a stub engine.
"""

import json
import os
import sys
import threading
import urllib.request

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import systemone.shim as shim
from systemone.shim import (
    build_route_question,
    candidates_from_registry,
    load_registry,
    parse_route_body,
    route_decision,
    serve,
)


@pytest.fixture()
def log_file(tmp_path, monkeypatch):
    """Point the decision log at a temp file; reset the logger singleton."""
    path = str(tmp_path / "decisions.log")
    monkeypatch.setenv("SYSTEMONE_LOG_FILE", path)
    monkeypatch.delenv("SYSTEMONE_LOG_DISABLE", raising=False)
    shim._logger = None
    yield path
    for h in list(getattr(shim._logger, "handlers", [])):
        try:
            h.close()
        except Exception:
            pass
    shim._logger = None


def _registry():
    return {
        "edge": {"model_id": "knowledgator/gliclass-edge-v3.0",
                 "description": "fastest, cheapest"},
        "base": {"model_id": "knowledgator/gliclass-base-v1.0",
                 "description": "slower, sharper"},
        "heavy": {"model_id": None, "description": "35B, unconfigured"},
    }


class StubEngine:
    model_name = "stub"

    def systemone(self, state, questions, batch_size=32):
        answers = {}
        for q in questions:
            opts = q["options"]
            n = len(opts)
            # deterministic: strongest weight on the first option
            raw = [n - i for i in range(n)]
            total = sum(raw)
            probs = {o: r / total for o, r in zip(opts, raw)}
            answers[q["name"]] = {
                "type": "choice", "choice": opts[0],
                "probabilities": probs, "confidence": probs[opts[0]],
            }
        answers["_meta"] = {"model": "stub"}
        return answers


# -- registry ---------------------------------------------------------------

def test_load_bundled_registry():
    reg = load_registry()
    assert reg["edge"]["model_id"] == "knowledgator/gliclass-edge-v3.0"
    assert reg["base"]["model_id"] == "knowledgator/gliclass-base-v1.0"
    assert reg["heavy"]["model_id"] is None  # unconfigured until user fills it in
    assert "latency_ms_p50" in reg["edge"]  # placeholder for measured receipts


def test_candidates_skips_unconfigured_tiers():
    cands = candidates_from_registry(_registry())
    assert [c["tier"] for c in cands] == ["edge", "base"]
    assert all(c["model_id"] for c in cands)


def test_candidates_subset_and_unknown_tier():
    cands = candidates_from_registry(_registry(), ["base"])
    assert [c["tier"] for c in cands] == ["base"]
    with pytest.raises(ValueError, match="unknown tier"):
        candidates_from_registry(_registry(), ["nope"])


def test_candidates_all_unconfigured_rejected():
    with pytest.raises(ValueError, match="no routable tiers"):
        candidates_from_registry(_registry(), ["heavy"])
    with pytest.raises(ValueError, match="model_id=null"):
        candidates_from_registry({"heavy": {"model_id": None}})


def test_candidates_invalid_model_id_type_rejected():
    with pytest.raises(ValueError, match="no routable tiers"):
        candidates_from_registry({"edge": {"model_id": 123}})


# -- request parsing ----------------------------------------------------------

def test_parse_route_body_defaults():
    task, bias, cands = parse_route_body({"task": "is this spam?"}, _registry())
    assert task == "is this spam?"
    assert bias == "balanced"
    assert [c["tier"] for c in cands] == ["edge", "base"]


def test_parse_route_body_rejects():
    with pytest.raises(ValueError, match="non-empty 'task'"):
        parse_route_body({}, _registry())
    with pytest.raises(ValueError, match="non-empty 'task'"):
        parse_route_body({"task": "   "}, _registry())
    with pytest.raises(ValueError, match="unknown cost_bias"):
        parse_route_body({"task": "x", "cost_bias": "turbo"}, _registry())
    with pytest.raises(ValueError, match="'tiers' must be a list"):
        parse_route_body({"task": "x", "tiers": "edge"}, _registry())


def test_build_route_question_shape():
    cands = candidates_from_registry(_registry())
    q = build_route_question("triage this ticket", cands, "economy")
    assert q["name"] == "route" and q["type"] == "choice"
    assert q["options"] == ["edge", "base"]
    assert "Aggressively prefer the cheapest" in q["prompt"]
    assert "knowledgator/gliclass-edge-v3.0" in q["prompt"]


def test_route_decision_picks_valid_registry_model():
    cands = candidates_from_registry(_registry())
    route = route_decision(StubEngine(), "summarize this contract", cands, "balanced")
    assert route["model_id"] in {c["model_id"] for c in cands}
    assert route["tier"] in {"edge", "base"}
    assert isinstance(route["rationale"], str) and len(route["rationale"]) > 20
    assert route["model_id"] in route["rationale"]  # rationale names the pick
    assert 0.0 < route["confidence"] <= 1.0
    assert abs(sum(route["probabilities"].values()) - 1.0) < 1e-9
    assert route["cost_bias"] == "balanced"


# -- HTTP ---------------------------------------------------------------------

def _serve_stub():
    server = serve(0, engine=StubEngine(), registry=_registry())
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, t


def _post(port, path, body):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.request.HTTPError as e:
        return e.code, json.loads(e.read())


def test_http_route_roundtrip(log_file):
    server, t = _serve_stub()
    try:
        status, payload = _post(
            server.server_address[1], "/v1/systemone/route",
            {"task": "triage this ticket", "cost_bias": "economy"},
        )
        assert status == 200
        route = payload["route"]
        assert route["model_id"] == "knowledgator/gliclass-edge-v3.0"
        assert route["tier"] == "edge"
        assert len(route["rationale"]) > 0
        assert payload["model"] == "stub"
        assert payload["usage"] == {}
        assert payload["latency_ms"] >= 0
    finally:
        server.shutdown()
        t.join(timeout=5)


def test_http_route_bad_requests():
    server, t = _serve_stub()
    port = server.server_address[1]
    try:
        for body in ({}, {"task": ""}, {"task": "x", "cost_bias": "turbo"},
                     {"task": "x", "tiers": ["nope"]},
                     {"task": "x", "tiers": ["heavy"]},
                     {"task": "x", "registry": {"heavy": {"model_id": None}}}):
            status, payload = _post(port, "/v1/systemone/route", body)
            assert status == 400, body
            assert "error" in payload
    finally:
        server.shutdown()
        t.join(timeout=5)


def test_http_route_custom_registry_inline():
    server, t = _serve_stub()
    try:
        status, payload = _post(
            server.server_address[1], "/v1/systemone/route",
            {"task": "x",
             "registry": {"tiny": {"model_id": "acme/tiny-1",
                                   "description": "smallest"}}},
        )
        assert status == 200
        assert payload["route"]["model_id"] == "acme/tiny-1"
        assert payload["route"]["tier"] == "tiny"
        assert len(payload["route"]["rationale"]) > 0
    finally:
        server.shutdown()
        t.join(timeout=5)


def _read_log(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_latency_logged_for_both_endpoints(log_file):
    server, t = _serve_stub()
    port = server.server_address[1]
    try:
        s1, _ = _post(port, "/v1/systemone",
                      {"state": "hi", "questions": {
                          "q": {"criteria": {"A": "a", "B": "b"}}}})
        s2, _ = _post(port, "/v1/systemone/route", {"task": "do a thing"})
        assert s1 == 200 and s2 == 200
    finally:
        server.shutdown()
        t.join(timeout=5)
    # flush/close handlers so the file is complete
    for h in list(shim._logger.handlers):
        h.flush()
    records = _read_log(log_file)
    by_endpoint = {r["endpoint"]: r for r in records}
    for endpoint in ("/v1/systemone", "/v1/systemone/route"):
        rec = by_endpoint[endpoint]
        assert rec["status"] == 200
        assert rec["latency_ms"] >= 0
        assert rec["model"] == "stub"
        assert "ts" in rec
    assert by_endpoint["/v1/systemone/route"]["route_tier"] == "edge"
    assert by_endpoint["/v1/systemone"]["n_questions"] == 1


def test_latency_logged_on_errors(log_file):
    server, t = _serve_stub()
    try:
        status, _ = _post(server.server_address[1], "/v1/systemone/route", {})
        assert status == 400
    finally:
        server.shutdown()
        t.join(timeout=5)
    for h in list(shim._logger.handlers):
        h.flush()
    records = _read_log(log_file)
    assert records and records[-1]["status"] == 400
    assert records[-1]["endpoint"] == "/v1/systemone/route"
