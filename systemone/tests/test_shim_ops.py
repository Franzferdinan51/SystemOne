"""Tests for shim ops: decision cache, batch endpoint, audit chain, PII.

Covers DecisionCache (TTL/LRU/disabled), POST /v1/systemone caching,
POST /v1/systemone/batch per-item semantics, the SYSTEMONE_AUDIT_CHAIN
hash chain (+ verify_audit_chain), and SYSTEMONE_SCRUB_PII redaction.
Stub engine + live HTTP; no model needed.
"""

import json
import os
import sys
import threading
import time
import urllib.request

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import systemone.shim as shim  # noqa: E402
from systemone.patterns import scrub_pii  # noqa: E402
from systemone.shim import (  # noqa: E402
    DecisionCache,
    decision_cache_from_env,
    log_decision,
    reset_audit_chain,
    serve,
    verify_audit_chain,
)


class StubEngine:
    model_name = "stub"

    def __init__(self):
        self.calls = 0
        self.states = []

    def systemone(self, state, questions, **kwargs):
        self.calls += 1
        self.states.append(state)
        answers = {}
        for q in questions:
            if q["type"] == "choice":
                opts = q["options"]
                answers[q["name"]] = {
                    "type": "choice", "choice": opts[0],
                    "probabilities": {o: 1.0 / len(opts) for o in opts},
                    "confidence": 0.5,
                }
            else:
                answers[q["name"]] = {
                    "type": "noul", "probability": 0.5, "answer": True,
                    "confidence": 0.0,
                }
        answers["_meta"] = {"model": "stub", "latency_ms": 3.0}
        return answers


def _body(state="the till is empty"):
    return {
        "model": "stub",
        "state": state,
        "questions": {"op": {"type": "choice",
                             "criteria": {"a": "A.", "b": "B."}}},
    }


def _post(port, path, body):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # nosec B310 -- test helper posting to a hardcoded 127.0.0.1 stub server; nosemgrep
            return resp.status, json.loads(resp.read())
    except urllib.request.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


def _serve(engine):
    server = serve(0, engine=engine)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, t


def _stop(server, t):
    server.shutdown()
    t.join(timeout=5)
    server.server_close()


# -- DecisionCache unit behavior -----------------------------------------


def test_cache_disabled_by_default():
    cache = DecisionCache()
    assert cache.enabled is False
    cache.put("k", {"a": 1})
    assert cache.get("k") is None


def test_cache_hit_miss_and_stats():
    cache = DecisionCache(ttl_seconds=60, max_entries=4)
    assert cache.get("k") is None
    cache.put("k", {"a": 1})
    assert cache.get("k") == {"a": 1}
    assert cache.stats() == {"enabled": True, "hits": 1, "misses": 1,
                             "size": 1}


def test_cache_expiry_and_lru():
    cache = DecisionCache(ttl_seconds=60, max_entries=2)
    cache.put("a", {"v": 1})
    cache.put("b", {"v": 2})
    cache.get("a")  # refresh a; b is now LRU
    cache.put("c", {"v": 3})
    assert cache.get("b") is None
    assert cache.get("a") == {"v": 1}
    # force-expire deterministically (no sleeping)
    with cache._lock:
        exp, payload = cache._entries["a"]
        cache._entries["a"] = (time.monotonic() - 1.0, payload)
    assert cache.get("a") is None


def test_cache_copies_isolate_mutations():
    cache = DecisionCache(ttl_seconds=60)
    cache.put("k", {"n": [1]})
    hit = cache.get("k")
    hit["n"].append(999)
    assert cache.get("k") == {"n": [1]}


def test_cache_from_env(monkeypatch):
    monkeypatch.delenv("SYSTEMONE_CACHE_TTL", raising=False)
    assert decision_cache_from_env().enabled is False
    monkeypatch.setenv("SYSTEMONE_CACHE_TTL", "30")
    assert decision_cache_from_env().enabled is True
    monkeypatch.setenv("SYSTEMONE_CACHE_TTL", "junk")
    assert decision_cache_from_env().enabled is False
    monkeypatch.setenv("SYSTEMONE_CACHE_TTL", "30")
    monkeypatch.setenv("SYSTEMONE_CACHE_MAX", "junk")
    assert decision_cache_from_env().enabled is False


# -- cache over HTTP ------------------------------------------------------


def test_systemone_cache_serves_repeats(monkeypatch):
    monkeypatch.setenv("SYSTEMONE_CACHE_TTL", "60")
    monkeypatch.setenv("SYSTEMONE_LOG_DISABLE", "1")
    engine = StubEngine()
    server, t = _serve(engine)
    try:
        port = server.server_address[1]
        body = _body()
        s1, p1 = _post(port, "/v1/systemone", body)
        s2, p2 = _post(port, "/v1/systemone", body)
        assert (s1, s2) == (200, 200)
        assert engine.calls == 1  # second request never touched it
        assert p1["cached"] is False
        assert p2["cached"] is True
        assert p2["latency_ms"] == 0.0
        assert p2["answers"] == p1["answers"]
        assert server.decision_cache.stats()["hits"] == 1
        # a different state is a different key
        s3, _ = _post(port, "/v1/systemone", _body("other till"))
        assert s3 == 200
        assert engine.calls == 2
    finally:
        _stop(server, t)


def test_systemone_no_cached_key_when_disabled(monkeypatch):
    monkeypatch.delenv("SYSTEMONE_CACHE_TTL", raising=False)
    monkeypatch.setenv("SYSTEMONE_LOG_DISABLE", "1")
    server, t = _serve(StubEngine())
    try:
        status, payload = _post(server.server_address[1], "/v1/systemone",
                                _body())
        assert status == 200
        assert "cached" not in payload
    finally:
        _stop(server, t)


# -- batch endpoint -------------------------------------------------------


def test_batch_mixed_results(monkeypatch):
    monkeypatch.setenv("SYSTEMONE_LOG_DISABLE", "1")
    engine = StubEngine()
    server, t = _serve(engine)
    try:
        status, payload = _post(
            server.server_address[1], "/v1/systemone/batch",
            {"items": [_body("first"), {"nope": True}, "junk", _body("second")]})
        assert status == 200
        assert payload["n_items"] == 4
        assert payload["model"] == "stub"
        codes = [r["status"] for r in payload["results"]]
        assert codes == [200, 400, 400, 200]
        assert "answers" in payload["results"][0]
        assert "error" in payload["results"][1]
        assert engine.calls == 2
    finally:
        _stop(server, t)


def test_batch_caps_and_validation(monkeypatch):
    monkeypatch.setenv("SYSTEMONE_LOG_DISABLE", "1")
    server, t = _serve(StubEngine())
    try:
        port = server.server_address[1]
        for bad in ({}, {"items": []}, {"items": "x"},
                     {"items": [_body()] * 33}):
            status, payload = _post(port, "/v1/systemone/batch", bad)
            assert status == 400, bad
            assert "error" in payload
    finally:
        _stop(server, t)


# -- audit chain ------------------------------------------------------------


@pytest.fixture()
def audit_log(tmp_path, monkeypatch):
    path = str(tmp_path / "audit.log")
    monkeypatch.setenv("SYSTEMONE_LOG_FILE", path)
    monkeypatch.delenv("SYSTEMONE_LOG_DISABLE", raising=False)
    monkeypatch.setenv("SYSTEMONE_AUDIT_CHAIN", "1")
    shim._logger = None
    reset_audit_chain()
    yield path
    for h in list(getattr(shim._logger, "handlers", [])):
        try:
            h.close()
        except Exception:
            pass
    shim._logger = None
    reset_audit_chain()


def test_audit_chain_verifies(audit_log):
    log_decision({"endpoint": "/x", "status": 200})
    log_decision({"endpoint": "/y", "status": 200})
    result = verify_audit_chain(audit_log)
    assert result == {"ok": True, "records": 2, "error": ""}
    rows = [json.loads(line) for line in open(audit_log).read().splitlines()]
    assert rows[0]["audit_seq"] == 1
    assert rows[1]["audit_prev"] == rows[0]["audit_hash"]


def test_audit_chain_detects_tampering(audit_log):
    log_decision({"endpoint": "/x", "status": 200})
    log_decision({"endpoint": "/y", "status": 200})
    lines = open(audit_log).read().splitlines()
    rec = json.loads(lines[1])
    rec["status"] = 500  # tamper after the fact
    lines[1] = json.dumps(rec)
    open(audit_log, "w").write("\n".join(lines) + "\n")
    result = verify_audit_chain(audit_log)
    assert result["ok"] is False
    assert "line 2" in result["error"]


def test_audit_chain_skips_unchained_lines(audit_log, monkeypatch):
    monkeypatch.delenv("SYSTEMONE_AUDIT_CHAIN")
    log_decision({"endpoint": "/plain", "status": 200})
    monkeypatch.setenv("SYSTEMONE_AUDIT_CHAIN", "1")
    log_decision({"endpoint": "/x", "status": 200})
    assert verify_audit_chain(audit_log)["records"] == 1


def test_verify_audit_chain_unreadable(tmp_path):
    result = verify_audit_chain(str(tmp_path / "missing.log"))
    assert result["ok"] is False


# -- PII scrubbing ----------------------------------------------------------


def test_scrub_pii_kinds():
    scrubbed, kinds = scrub_pii(
        "mail bob@x.com, call 415-555-0132, ssn 123-45-6789")
    assert "[REDACTED_EMAIL]" in scrubbed
    assert "bob@x.com" not in scrubbed
    assert kinds == ["email", "phone", "ssn"]
    assert scrub_pii("nothing sensitive here") == (
        "nothing sensitive here", [])
    assert scrub_pii("") == ("", [])


def test_systemone_scrubs_state_when_enabled(monkeypatch):
    monkeypatch.setenv("SYSTEMONE_SCRUB_PII", "1")
    monkeypatch.setenv("SYSTEMONE_LOG_DISABLE", "1")
    engine = StubEngine()
    server, t = _serve(engine)
    try:
        status, payload = _post(
            server.server_address[1], "/v1/systemone",
            _body("contact bob@x.com about the till"))
        assert status == 200
        assert payload["pii"]["redacted"] is True
        assert payload["pii"]["kinds"] == ["email"]
        assert payload["pii"]["count"] == 1
        assert "bob@x.com" not in engine.states[0]
        assert "[REDACTED_EMAIL]" in engine.states[0]
    finally:
        _stop(server, t)


def test_systemone_no_scrub_by_default(monkeypatch):
    monkeypatch.delenv("SYSTEMONE_SCRUB_PII", raising=False)
    monkeypatch.setenv("SYSTEMONE_LOG_DISABLE", "1")
    engine = StubEngine()
    server, t = _serve(engine)
    try:
        status, payload = _post(
            server.server_address[1], "/v1/systemone",
            _body("contact bob@x.com about the till"))
        assert status == 200
        assert "pii" not in payload
        assert "bob@x.com" in engine.states[0]
    finally:
        _stop(server, t)
