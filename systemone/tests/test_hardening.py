"""Tests for request hardening: body caps, batch caps, optional auth.

No model needed — pure-function checks plus stub engines over real HTTP.
"""

import json
import os
import sys
import threading
import urllib.error
import urllib.request

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from systemone import patterns  # noqa: E402
from systemone.patterns import (  # noqa: E402
    MAX_BODY_BYTES,
    BodyTooLarge,
    api_token_ok,
    check_body_length,
)
from systemone.shim import serve  # noqa: E402


class StubEngine:
    model_name = "stub"

    def systemone(self, state, questions):
        answers = {}
        for q in questions:
            if q["type"] == "choice":
                opts = q["options"]
                answers[q["name"]] = {
                    "type": "choice", "choice": opts[0],
                    "probabilities": {o: 1.0 / len(opts) for o in opts},
                    "confidence": 0.5,
                }
            elif q["type"] == "score":
                lv = q["levels"]
                answers[q["name"]] = {
                    "type": "score", "level": lv[0],
                    "distribution": {x: 1.0 / len(lv) for x in lv},
                    "confidence": 0.5,
                }
            else:
                answers[q["name"]] = {
                    "type": "noul", "probability": 0.8, "answer": True,
                    "confidence": 0.8,
                }
        answers["_meta"] = {"model": "stub", "latency_ms": 1.0}
        return answers


# -- pure checks --------------------------------------------------------


def test_check_body_length_bounds():
    assert MAX_BODY_BYTES == 2 * 1024 * 1024
    assert check_body_length({"Content-Length": "12"}) == 12
    assert check_body_length({}) == 0
    assert check_body_length(None) == 0
    with pytest.raises(ValueError):
        check_body_length({"Content-Length": "junk"})
    with pytest.raises(ValueError):
        check_body_length({"Content-Length": "-5"})
    with pytest.raises(BodyTooLarge):
        check_body_length({"Content-Length": str(MAX_BODY_BYTES + 1)})
    # BodyTooLarge must NOT be a ValueError (else it maps to 400, not 413).
    assert not issubclass(BodyTooLarge, ValueError)


def test_api_token_gate(monkeypatch):
    monkeypatch.delenv("SYSTEMONE_API_TOKEN", raising=False)
    assert api_token_ok(None) is True
    assert api_token_ok("Bearer anything") is True
    monkeypatch.setenv("SYSTEMONE_API_TOKEN", "s3cret")
    assert api_token_ok("Bearer s3cret") is True
    assert api_token_ok(None) is False
    assert api_token_ok("Bearer wrong") is False
    assert api_token_ok("s3cret") is False


# -- HTTP ---------------------------------------------------------------


def _post(url, body, headers=None):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _serve(engine):
    server = serve(0, engine=engine)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, t


def _close(server, t):
    server.shutdown()
    t.join(timeout=5)
    server.server_close()


def test_huge_body_rejected_with_413(monkeypatch):
    monkeypatch.setattr(patterns, "MAX_BODY_BYTES", 64)
    server, t = _serve(StubEngine())
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        status, payload = _post(f"{base}/v1/systemone", {
            "state": "x" * 1000, "questions": {"q": {"type": "noul"}}})
        assert status == 413
        assert "too large" in payload["error"]
    finally:
        _close(server, t)


def test_decisions_question_cap_is_422():
    server, t = _serve(StubEngine())
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        body = {"input": "s", "questions": [
            {"id": f"q{i}", "type": "yes_no", "question": "Q?"}
            for i in range(65)]}
        status, _ = _post(f"{base}/v1/decisions", body)
        assert status == 422
        body["questions"] = body["questions"][:64]
        status, _ = _post(f"{base}/v1/decisions", body)
        assert status == 200
    finally:
        _close(server, t)


def test_systemone_question_cap_is_400():
    server, t = _serve(StubEngine())
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        body = {"state": "s", "questions": {
            f"q{i}": {"type": "noul"} for i in range(65)}}
        status, _ = _post(f"{base}/v1/systemone", body)
        assert status == 400
    finally:
        _close(server, t)


def test_rank_plans_cap_is_400():
    server, t = _serve(StubEngine())
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        body = {"task": "t", "plans": [
            {"id": f"p{i}", "text": "do it"} for i in range(33)]}
        status, _ = _post(f"{base}/v1/systemone/rank-plans", body)
        assert status == 400
    finally:
        _close(server, t)


def test_auth_gate_401_and_200(monkeypatch):
    monkeypatch.setenv("SYSTEMONE_API_TOKEN", "s3cret")
    server, t = _serve(StubEngine())
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        body = {"state": "s", "questions": {"q": {"type": "noul"}}}
        status, _ = _post(f"{base}/v1/systemone", body)
        assert status == 401
        status, _ = _post(f"{base}/v1/systemone", body,
                           {"Authorization": "Bearer wrong"})
        assert status == 401
        status, _ = _post(f"{base}/v1/systemone", body,
                           {"Authorization": "Bearer s3cret"})
        assert status == 200
        # Health probes stay open under a token.
        with urllib.request.urlopen(f"{base}/healthz", timeout=10) as resp:
            assert resp.status == 200
    finally:
        _close(server, t)


def test_auth_open_by_default(monkeypatch):
    monkeypatch.delenv("SYSTEMONE_API_TOKEN", raising=False)
    server, t = _serve(StubEngine())
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        status, _ = _post(f"{base}/v1/systemone", {
            "state": "s", "questions": {"q": {"type": "noul"}}})
        assert status == 200
    finally:
        _close(server, t)


# -- observability ----------------------------------------------------------


def test_metrics_counts_requests_by_endpoint_and_status():
    server, t = _serve(StubEngine())
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        _post(f"{base}/v1/systemone", {
            "state": "s", "questions": {"q": {"type": "noul"}}})
        _post(f"{base}/v1/systemone", {"bogus": True})
        with urllib.request.urlopen(f"{base}/healthz", timeout=10):
            pass
        try:
            urllib.request.urlopen(f"{base}/nope", timeout=10)
        except urllib.error.HTTPError as e:
            assert e.code == 404
        with urllib.request.urlopen(f"{base}/metrics", timeout=10) as resp:
            assert resp.status == 200
            assert resp.headers.get_content_type() == "text/plain"
            text = resp.read().decode()
        assert 'systemone_requests_total{endpoint="/v1/systemone",status="200"} 1' in text
        assert 'systemone_requests_total{endpoint="/v1/systemone",status="400"} 1' in text
        assert 'systemone_requests_total{endpoint="/healthz",status="200"} 1' in text
        assert 'systemone_requests_total{endpoint="/nope",status="404"} 1' in text
        assert "systemone_request_latency_ms_sum" in text
    finally:
        _close(server, t)


def test_request_id_echoed_and_generated():
    server, t = _serve(StubEngine())
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        req = urllib.request.Request(
            f"{base}/healthz", headers={"X-Request-ID": "trace-1"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.headers["X-Request-ID"] == "trace-1"
        with urllib.request.urlopen(f"{base}/healthz", timeout=10) as resp:
            generated = resp.headers["X-Request-ID"]
        assert generated and generated != "-"
        with urllib.request.urlopen(f"{base}/healthz", timeout=10) as resp:
            assert resp.headers["X-Request-ID"] != generated
    finally:
        _close(server, t)
