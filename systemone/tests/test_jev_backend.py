"""Tests for the JEV decision-model backend (hosted /v1/decide + vllm-raw).

All HTTP is mocked — no server, weights, or downloads needed.
"""

import json
import os
import sys
import urllib.error
import urllib.request

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from systemone import JevDecideBackend, JevError  # noqa: E402
from systemone.jev_backend import build_state, image_part  # noqa: E402
from systemone.shim import create_engine, engine_backend_name  # noqa: E402


class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status = status

    def read(self):
        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _hosted_ok(body):
    assert body["kind"] == "choice"
    assert body["options"] == ["a", "b"]
    return {
        "kind": "choice", "effective_kind": "choice",
        "options": ["a", "b"], "probabilities": [0.8, 0.2],
        "choice_index": 0, "choice": "a", "adaptation": "native",
        "protocol": "jev27-bare-v1", "model": "autotrust/JEV-27B-VL",
        "usage": {"prompt_tokens": 49, "completion_tokens": 1,
                  "total_tokens": 50},
        "num_model_requests": 1, "elapsed_seconds": 0.02,
    }


def test_decide_hosted_returns_jev_response(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["headers"] = dict(req.header_items())
        body = json.loads(req.data.decode())
        return _FakeResp(_hosted_ok(body))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    eng = JevDecideBackend(base_url="http://jev-test:8000")
    resp = eng.decide("choice", "some state", "Which?", ["a", "b"])
    assert seen["url"] == "http://jev-test:8000/v1/decide"
    assert "Authorization" not in seen["headers"]
    assert resp["choice"] == "a"
    assert resp["probabilities"] == [0.8, 0.2]
    assert resp["num_model_requests"] == 1
    assert eng.model_name == "autotrust/JEV-27B-VL"


def test_decide_sends_bearer_key(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["headers"] = dict(req.header_items())
        return _FakeResp(_hosted_ok(json.loads(req.data.decode())))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    eng = JevDecideBackend(base_url="http://jev-test:8000", api_key="sk-test")
    eng.decide("choice", "s", "Q?", ["a", "b"])
    assert seen["headers"]["Authorization"] == "Bearer sk-test"


def test_decide_validates_kind_options_question():
    eng = JevDecideBackend(base_url="http://jev-test:8000")
    with pytest.raises(JevError):
        eng.decide("bogus", "s", "Q?", ["a", "b"])
    with pytest.raises(JevError):
        eng.decide("choice", "s", "Q?", ["only-one"])
    with pytest.raises(JevError):
        eng.decide("choice", "s", "  ", ["a", "b"])


def test_decide_connection_failure_hint(monkeypatch):
    def boom(req, timeout=None):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    eng = JevDecideBackend(base_url="http://jev-test:8000")
    with pytest.raises(JevError) as ei:
        eng.decide("choice", "s", "Q?", ["a", "b"])
    assert "JEV_URL" in str(ei.value.hint)


def test_decide_malformed_reply_raises(monkeypatch):
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda req, timeout=None: _FakeResp({"oops": True}),
    )
    eng = JevDecideBackend(base_url="http://jev-test:8000")
    with pytest.raises(JevError):
        eng.decide("choice", "s", "Q?", ["a", "b"])


def test_systemone_maps_all_types(monkeypatch):
    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data.decode())
        kind = body["kind"]
        if kind == "noul":
            opts, probs = ["false", "true"], [0.25, 0.75]
        elif body.get("options"):
            opts = body["options"]
            probs = [0.7] + [0.3 / (len(opts) - 1)] * (len(opts) - 1)
        else:
            opts = [str(i) for i in range(6)]
            probs = [0.05, 0.05, 0.1, 0.2, 0.25, 0.35]
        return _FakeResp({
            "options": opts, "probabilities": probs,
            "choice_index": probs.index(max(probs)),
            "choice": opts[probs.index(max(probs))],
        })

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    eng = JevDecideBackend(base_url="http://jev-test:8000")
    out = eng.systemone("state", [
        {"name": "action", "type": "choice", "options": ["up", "down"],
         "prompt": "Move?"},
        {"name": "level", "type": "score", "levels": ["0", "1", "2", "3", "4", "5"]},
        {"name": "custom", "type": "score", "levels": ["low", "high"]},
        {"name": "ok", "type": "noul", "statement": "It is fine."},
    ])
    assert out["action"]["choice"] == "up"
    assert out["action"]["confidence"] > 0
    assert out["_meta"]["backend"] == "jev"
    assert out["level"]["level"] == "5"
    assert out["custom"]["level"] == "low"  # custom levels via choice
    assert out["ok"]["answer"] is True
    assert out["ok"]["probability"] == pytest.approx(0.75)


def test_systemone_noul_needs_statement():
    eng = JevDecideBackend(base_url="http://jev-test:8000")
    with pytest.raises(JevError):
        eng.systemone("s", [{"name": "x", "type": "noul"}])


def test_images_travel_in_state(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["body"] = json.loads(req.data.decode())
        return _FakeResp(_hosted_ok({"kind": "choice", "options": ["a", "b"]}))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    eng = JevDecideBackend(base_url="http://jev-test:8000")
    eng.decide("choice", "Photo: ", "Which?", ["a", "b"],
               images=["https://example.com/p.jpg"])
    state = seen["body"]["state"]
    assert isinstance(state, list)
    assert {"image": "https://example.com/p.jpg"} in state


def test_image_part_normalizes_openai_form():
    assert image_part({"type": "image_url",
                       "image_url": {"url": "https://e.com/x.png"}}) == {
        "image": "https://e.com/x.png"}
    assert image_part({"image": "data:x"}) == {"image": "data:x"}
    with pytest.raises(JevError):
        image_part({"type": "text", "text": "nope"})


def test_build_state_text_only_stays_scalar():
    assert build_state("hello") == "hello"
    assert build_state("hello", ["https://e.com/x.png"]) == [
        "hello", {"image": "https://e.com/x.png"}]


def test_build_state_json_and_lists_pass_through():
    assert build_state({"a": 1}) == {"a": 1}
    assert build_state(["a", "b"]) == ["a", "b"]
    assert build_state(None) == ""
    assert build_state({"a": 1}, ["https://e.com/x.png"]) == [
        {"a": 1}, {"image": "https://e.com/x.png"}]


def test_decide_derives_choice_from_index(monkeypatch):
    def fake_urlopen(req, timeout=None):
        return _FakeResp({
            "options": ["a", "b"], "probabilities": [0.8, 0.2],
            "choice_index": 0, "choice": "b",  # disagreeing label
        })

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    eng = JevDecideBackend(base_url="http://jev-test:8000")
    resp = eng.decide("choice", "s", "Q?", ["a", "b"])
    assert resp["choice"] == "a"
    assert resp["options"][resp["choice_index"]] == resp["choice"]


def test_systemone_empty_score_levels_raise():
    eng = JevDecideBackend(base_url="http://jev-test:8000")
    with pytest.raises(JevError, match=">= 2 levels"):
        eng.systemone("s", [{"name": "x", "type": "score", "levels": []}])


def test_score_confidence_uses_typesafe_helper(monkeypatch):
    from systemone.patterns import score_confidence

    def fake_urlopen(req, timeout=None):
        return _FakeResp({
            "options": ["0", "1", "2"], "probabilities": [0.1, 0.2, 0.7],
            "choice_index": 2, "choice": "2",
        })

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    eng = JevDecideBackend(base_url="http://jev-test:8000")
    out = eng.systemone(
        "s", [{"name": "x", "type": "score",
               "levels": ["low", "mid", "high"]}])
    assert out["x"]["confidence"] == score_confidence([0.1, 0.2, 0.7])


def test_vllm_raw_path_uses_bundle_math(monkeypatch, tmp_path):
    bundle = tmp_path / "b"
    (bundle / "adapter_vllm").mkdir(parents=True)
    (bundle / "adapter_vllm" / "decision_head.json").write_text(json.dumps({
        "slots": {"ranges": {"choice": [0]}},
        "verbalizer_ids": [10, 11],
        "bias": [0.0, 0.0],
    }))
    (bundle / "calibration.json").write_text(json.dumps(
        {"per_kind": {"choice": 1.0}}))
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        body = json.loads(req.data.decode())
        seen["body"] = body
        return _FakeResp({"choices": [{"logprobs": {"top_logprobs": [{
            "10:10": -0.2, "11:11": -1.5}]}}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    eng = JevDecideBackend(base_url="http://jev-test:8000", backend="vllm",
                           bundle_dir=str(bundle))
    resp = eng.decide("choice", "s", "Q?", ["a", "b"])
    assert seen["url"] == "http://jev-test:8000/v1/completions"
    assert seen["body"]["top_k"] == 0 and seen["body"]["top_p"] == 1.0
    assert seen["body"]["max_tokens"] == 1
    assert resp["choice"] == "a"
    assert resp["adaptation"] == "client-math"
    assert abs(sum(resp["probabilities"]) - 1.0) < 1e-9


def test_vllm_raw_path_needs_bundle():
    eng = JevDecideBackend(base_url="http://jev-test:8000", backend="vllm")
    with pytest.raises(JevError) as ei:
        eng.decide("choice", "s", "Q?", ["a", "b"])
    assert "JEV_BUNDLE" in str(ei.value.hint)


def test_chat_returns_answer_text(monkeypatch):
    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data.decode())
        assert body["chat_template_kwargs"] == {"enable_thinking": True}
        return _FakeResp({"choices": [{"message": {
            "content": "<think>hmm</think>do the thing"}}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    eng = JevDecideBackend(base_url="http://jev-test:8000")
    assert eng.chat("What next?", thinking=True) == "do the thing"


def test_engine_names_jev_and_auto_prefers_it(monkeypatch):
    monkeypatch.setattr(JevDecideBackend, "health", lambda self: True)
    monkeypatch.delenv("SGLANG_BASE_URL", raising=False)
    monkeypatch.delenv("JEVK5_BASE_URL", raising=False)
    monkeypatch.setenv("JEV_URL", "http://jev-test:8000")
    eng = create_engine("jev")
    assert engine_backend_name(eng) == "jev"
    auto = create_engine("auto")
    assert engine_backend_name(auto) == "jev"
    assert "jev" in str(create_engine.__doc__)


def test_bad_backend_name_rejected():
    with pytest.raises(JevError):
        JevDecideBackend(base_url="http://jev-test:8000", backend="bogus")


def test_non_http_url_rejected():
    with pytest.raises(Exception):
        JevDecideBackend(base_url="ftp://jev-test:8000")
