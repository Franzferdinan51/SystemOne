"""Tests for the SGLang backend. No server, no model — urlopen is mocked."""

import io
import json
import os
import sys
import urllib.error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from systemone.sglang_backend import (
    SGLangBackend,
    SGLangError,
    _check_distribution,
)


class FakeResp:
    def __init__(self, payload, status=200):
        self._data = json.dumps(payload).encode()
        self.status = status

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _backend(monkeypatch, payload=None, fail=None):
    """SGLangBackend with urlopen mocked. `fail` may be an exception to raise."""
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode())
        captured["timeout"] = timeout
        if fail is not None:
            raise fail
        return FakeResp(payload or {"answers": {}})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    b = SGLangBackend(base_url="http://sglang-test:30000", timeout=5)
    return b, captured


def _answers_payload():
    return {
        "answers": {
            "team": {
                "type": "choice",
                "choice": "backend",
                "probabilities": {"backend": 0.7, "frontend": 0.2, "devops": 0.1},
                "label_mass": 0.93,
            },
            "impact": {
                "type": "score",
                "score": 2.4,
                "probabilities": {"low": 0.1, "medium": 0.4, "high": 0.5},
                "label_mass": 0.88,
            },
            "page": {
                "type": "yes_no",
                "probability": 0.72,
                "answer": True,
                "label_mass": 0.91,
            },
        }
    }


def _questions():
    return [
        {"name": "team", "type": "choice",
         "options": ["backend", "frontend", "devops"]},
        {"name": "impact", "type": "score",
         "levels": ["low", "medium", "high"]},
        {"name": "page", "type": "noul",
         "statement": "Should the CTO be paged?"},
    ]


def test_request_schema_matches_sglang_decisions(monkeypatch):
    b, captured = _backend(monkeypatch, _answers_payload())
    b.systemone("the server is on fire", _questions())
    assert captured["url"] == "http://sglang-test:30000/v1/decisions"
    body = captured["body"]
    assert body["input"] == "the server is on fire"
    by_id = {q["id"]: q for q in body["questions"]}
    assert by_id["team"]["type"] == "choice"
    assert [o["name"] for o in by_id["team"]["options"]] == [
        "backend", "frontend", "devops"]
    assert by_id["impact"]["type"] == "score"
    assert [lv["name"] for lv in by_id["impact"]["levels"]] == [
        "low", "medium", "high"]
    assert by_id["page"]["type"] == "yes_no"
    assert "question" in by_id["page"]


def test_answer_mapping_and_meta(monkeypatch):
    b, captured = _backend(monkeypatch, _answers_payload())
    out = b.systemone("state", _questions())
    assert out["team"]["choice"] == "backend"
    assert out["team"]["confidence"] == pytest.approx(0.7)
    assert out["team"]["label_mass"] == pytest.approx(0.93)
    assert out["impact"]["type"] == "score"
    assert out["impact"]["level"] == "high"
    assert out["impact"]["score"] == pytest.approx(2.4)
    assert out["page"]["type"] == "noul"
    assert out["page"]["answer"] is True
    assert out["page"]["probability"] == pytest.approx(0.72)
    meta = out["_meta"]
    assert meta["backend"] == "sglang"
    assert meta["n_questions"] == 3
    assert "latency_ms" in meta


def test_too_many_options_refused_client_side(monkeypatch):
    b, _ = _backend(monkeypatch, {"answers": {}})
    with pytest.raises(SGLangError):
        b.systemone("s", [{"name": "x", "type": "choice",
                           "options": [f"o{i}" for i in range(27)]}])


def test_duplicate_ids_rejected(monkeypatch):
    b, _ = _backend(monkeypatch, {"answers": {}})
    qs = [{"name": "dup", "type": "noul", "statement": "a?"},
          {"name": "dup", "type": "noul", "statement": "b?"}]
    with pytest.raises(SGLangError):
        b.systemone("s", qs)


def test_http_error_becomes_sanitized_sglang_error(monkeypatch):
    err = urllib.error.HTTPError(
        "http://x/v1/decisions", 500, "boom", {}, io.BytesIO(b"{}"))
    b, _ = _backend(monkeypatch, fail=err)
    with pytest.raises(SGLangError, match="HTTP 500"):
        b.systemone("s", _questions())


def test_connection_refused_becomes_sglang_error(monkeypatch):
    b, _ = _backend(monkeypatch,
                    fail=urllib.error.URLError("connection refused"))
    with pytest.raises(SGLangError, match="could not reach"):
        b.systemone("s", _questions())


def test_invalid_server_distribution_rejected(monkeypatch):
    bad = {"answers": {"team": {
        "type": "choice", "choice": "backend",
        "probabilities": {"backend": 0.7, "frontend": 0.2},  # missing devops
        "label_mass": 0.9}}}
    b, _ = _backend(monkeypatch, bad)
    with pytest.raises(SGLangError, match="invalid decision distribution"):
        b.systemone("s", _questions())


def test_check_distribution_contract():
    good = {"a": 0.6, "b": 0.4}
    assert _check_distribution(good, ["a", "b"], "a") == good
    with pytest.raises(SGLangError):
        _check_distribution({"a": 0.6, "b": 0.4}, ["a", "b"], "b")
    with pytest.raises(SGLangError):
        # sums to 0.999, outside a tight tolerance
        _check_distribution({"a": 0.499, "b": 0.5}, ["a", "b"], "b", tol=1e-9)


def test_speculative_decide_uses_only_chosen_head(monkeypatch):
    payload = {"answers": {
        "op": {"type": "choice", "choice": "click",
               "probabilities": {"click": 0.8, "wait": 0.2}, "label_mass": 0.9},
        "click_target": {"type": "choice", "choice": "btn-2",
                         "probabilities": {"btn-1": 0.3, "btn-2": 0.7},
                         "label_mass": 0.9},
        "wait_target": {"type": "choice", "choice": "y",
                        "probabilities": {"x": 0.2, "y": 0.8},
                        "label_mass": 0.9},
    }}
    b, _ = _backend(monkeypatch, payload)
    out = b.speculative_decide(
        "state",
        {"name": "op", "type": "choice", "options": ["click", "wait"]},
        {"click": {"name": "click_target", "type": "choice",
                   "options": ["btn-1", "btn-2"]},
         "wait": {"name": "wait_target", "type": "choice",
                  "options": ["x", "y"]}},
    )
    assert out["operation"] == "click"
    assert out["target"] == "btn-2"  # "y" would mean the wrong head was used
    assert out["target_confidence"] == pytest.approx(0.7)


def test_env_config(monkeypatch):
    monkeypatch.setenv("SGLANG_BASE_URL", "http://gpu-box:30000/")
    monkeypatch.setenv("SGLANG_MODEL", "Qwen/Qwen3.8-27B")
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode())
        return FakeResp(_answers_payload())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    b = SGLangBackend()
    assert b.base_url == "http://gpu-box:30000"  # trailing slash stripped
    assert b.model_name == "Qwen/Qwen3.8-27B"
    b.systemone("s", _questions())
    assert captured["body"]["model"] == "Qwen/Qwen3.8-27B"
