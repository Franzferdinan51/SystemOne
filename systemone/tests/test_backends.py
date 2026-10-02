"""Tests for the beyond-GLiClass backends: rerank, ONNX, JevK5-server, Kev.

RerankBackend, JevK5ServerBackend, and KevBackend are pure logic + mocked
HTTP (always run). The OnnxCrossEncoder integration runs only when
onnxruntime and a cached model are present — tests never download weights.
"""

import importlib.util
import json
import os
import sys
import urllib.request

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from systemone import (  # noqa: E402
    JevK5Error,
    JevK5ServerBackend,
    KevBackend,
    KevError,
    OnnxCrossEncoder,
    RerankBackend,
)
from systemone import shim as shim_module  # noqa: E402
from systemone.shim import create_engine, engine_backend_name  # noqa: E402


def _questions():
    return [
        {"name": "action", "type": "choice",
         "options": ["up", "down"], "prompt": "Move?",
         "descriptions": {"up": "go up", "down": "go down"}},
        {"name": "level", "type": "score", "levels": ["low", "high"]},
        {"name": "ok", "type": "noul", "statement": "It is fine."},
    ]


def test_rerank_maps_all_types_with_confidence():
    seen = []

    def score_fn(query, doc):
        seen.append((query, doc))
        return 2.0 if doc.startswith("up") or doc in ("high", "It is fine.") else 0.0

    out = RerankBackend(score_fn, model_name="fake").systemone("s", _questions())
    assert out["action"]["choice"] == "up"
    assert out["action"]["probabilities"]["up"] > 0.5
    assert 0.0 <= out["action"]["confidence"] <= 1.0
    assert out["level"]["level"] == "high"
    assert out["level"]["distribution"]["high"] > 0.5
    assert out["ok"]["probability"] > 0.5 and out["ok"]["answer"] is True
    assert out["_meta"]["backend"] == "rerank"
    # descriptions reach the pair text; instructions reach the query
    assert any("go up" in doc for _, doc in seen)
    assert any("Move?" in q for q, _ in seen)
    assert all("Options:" not in q for q, _ in seen)  # block stripped


def test_rerank_rejects_bad_questions():
    eng = RerankBackend(lambda q, d: 0.0)
    try:
        eng.systemone("s", [{"name": "a", "type": "choice", "options": ["x"]}])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for 1 option")
    try:
        eng.systemone("s", [{"name": "a", "type": "mystery"}])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown type")


def test_rerank_noul_temperature_shapes_probability():
    eng_hot = RerankBackend(lambda q, d: 2.0, noul_temperature=0.5)
    eng_cold = RerankBackend(lambda q, d: 2.0, noul_temperature=8.0)
    p_hot = eng_hot.systemone("s", [q for q in _questions()
                                    if q["type"] == "noul"])["ok"]["probability"]
    p_cold = eng_cold.systemone("s", [q for q in _questions()
                                      if q["type"] == "noul"])["ok"]["probability"]
    assert p_hot > p_cold > 0.5


class _FakeResp:
    status = 200

    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_jevk5_posts_typesafe_dialect(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data.decode())
        return _FakeResp({"answers": {
            "action": {"choice": "up",
                       "probabilities": {"up": 0.8, "down": 0.2}},
            "level": {"level": "high",
                      "distribution": {"low": 0.3, "high": 0.7}},
            "ok": {"noul": 0.9},
        }})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    out = JevK5ServerBackend(base_url="http://jevk5-test:8090").systemone(
        "s", _questions())
    assert seen["url"] == "http://jevk5-test:8090/v1/systemone"
    sent = seen["body"]["questions"]["action"]
    assert sent["type"] == "choice" and set(sent["criteria"]) == {"up", "down"}
    assert out["action"]["choice"] == "up"
    assert out["level"]["level"] == "high"
    assert out["ok"]["probability"] == 0.9
    assert out["_meta"]["backend"] == "jevk5"


def test_jevk5_health_tries_health_then_healthz(monkeypatch):
    urls = []

    def fake_urlopen(req, timeout=None):
        urls.append(req.full_url)
        if req.full_url.endswith("/health"):
            raise ValueError("nobody home")
        return _FakeResp({"ok": True})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert JevK5ServerBackend(base_url="http://jevk5-test:8090").health() is True
    assert urls[0].endswith("/health") and urls[1].endswith("/healthz")


def test_jevk5_missing_answer_raises(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=None: _FakeResp({"answers": {}}))
    try:
        JevK5ServerBackend(base_url="http://jevk5-test:8090").systemone(
            "s", _questions()[:1])
    except JevK5Error as exc:
        assert "missing answer" in str(exc)
    else:
        raise AssertionError("expected JevK5Error")


def test_jevk5_rejects_non_http():
    try:
        JevK5ServerBackend(base_url="file:///etc/passwd")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for file:// URL")


def test_engine_names_jevk5_and_labels(monkeypatch):
    monkeypatch.setattr(JevK5ServerBackend, "health", lambda self: True)
    monkeypatch.delenv("SGLANG_BASE_URL", raising=False)
    eng = create_engine("jevk5")
    assert isinstance(eng, JevK5ServerBackend)
    assert engine_backend_name(eng) == "jevk5"
    assert engine_backend_name(RerankBackend(lambda q, d: 0.0)) == "rerank"
    monkeypatch.setenv("JEVK5_BASE_URL", "http://jevk5-test:8090")
    assert isinstance(create_engine("auto"), JevK5ServerBackend)


def test_engine_onnx_uses_env_model(monkeypatch):
    made = {}

    class FakeEnc:
        model_id = "fake-model"

        def __init__(self, model_id=None, filename=None, **kw):
            made["model_id"] = model_id
            made["filename"] = filename

        def score(self, q, d):
            return 0.0

    monkeypatch.setattr(shim_module, "OnnxCrossEncoder", FakeEnc)
    monkeypatch.setenv("RERANK_MODEL_ID", "my-model")
    eng = create_engine("onnx")
    assert isinstance(eng, RerankBackend)
    assert made == {"model_id": "my-model", "filename": "onnx/model_int8.onnx"}


def _onnx_ready():
    for mod in ("onnxruntime", "tokenizers", "huggingface_hub"):
        if importlib.util.find_spec(mod) is None:
            return False
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download("Xenova/bge-reranker-base", "onnx/model_int8.onnx",
                        local_files_only=True)
        hf_hub_download("Xenova/bge-reranker-base", "tokenizer.json",
                        local_files_only=True)
        return True
    except Exception:
        return False


needs_onnx = pytest.mark.skipif(
    not _onnx_ready(), reason="needs cached ONNX reranker (no downloads in tests)")


@needs_onnx
def test_onnx_encoder_ranks_paraphrase_first():
    enc = OnnxCrossEncoder()
    good = enc.score("the sky is blue", "the sky is blue in color")
    bad = enc.score("the sky is blue", "quantum banana firmware protocol")
    assert good > bad


# -- KevBackend (kev.serve client) ------------------------------------------


def _kev_payload():
    # Real kev.serve shapes (kev/api.py to_answers): noul carries only
    # "noul"; score probabilities are index-keyed with a legend.
    return {"answers": {
        "action": {"type": "choice", "choice": "up", "confidence": 0.6,
                   "probabilities": {"up": 0.8, "down": 0.2}},
        "level": {"type": "score", "score": 0.7, "confidence": 0.4,
                  "legend": {"0": "low", "1": "high"},
                  "probabilities": {"0": 0.3, "1": 0.7}},
        "ok": {"type": "noul", "noul": 0.9},
    }, "usage": {"input_tokens": 42, "output_tokens": 17}}


def test_kev_posts_typesafe_dialect(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data.decode())
        return _FakeResp(_kev_payload())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    out = KevBackend(base_url="http://kev-test:8008").systemone(
        "s", _questions())
    assert seen["url"] == "http://kev-test:8008/v1/systemone"
    assert seen["body"]["model"] == "kev-latest"
    sent = seen["body"]["questions"]["action"]
    assert sent["type"] == "choice" and set(sent["criteria"]) == {
        "up", "down"}
    assert out["action"]["choice"] == "up"
    assert out["level"]["level"] == "high"
    assert out["level"]["distribution"] == {"low": 0.3, "high": 0.7}
    assert out["level"]["legend"] == {"low": "low", "high": "high"}
    assert out["ok"]["probability"] == 0.9
    assert out["_meta"]["backend"] == "kev"
    assert out["_meta"]["usage"] == {"input_tokens": 42,
                                     "output_tokens": 17}


def test_kev_sends_bearer_auth_when_configured(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["auth"] = req.get_header("Authorization")
        return _FakeResp(_kev_payload())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("KEV_API_KEY", "s3cret")
    KevBackend(base_url="http://kev-test:8008").systemone(
        "s", _questions()[:1])
    assert seen["auth"] == "Bearer s3cret"


def test_kev_health_reads_v1_models(monkeypatch):
    urls = []

    def fake_urlopen(req, timeout=None):
        urls.append(req.full_url)
        return _FakeResp({"models": [{"name": "kev-latest"}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert KevBackend(base_url="http://kev-test:8008").health() is True
    assert urls == ["http://kev-test:8008/v1/models"]


def test_kev_health_false_on_transport_error(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise ValueError("nobody home")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert KevBackend(base_url="http://kev-test:8008").health() is False


def test_kev_missing_answer_raises(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=None: _FakeResp({"answers": {}}))
    try:
        KevBackend(base_url="http://kev-test:8008").systemone(
            "s", _questions()[:1])
    except KevError as exc:
        assert "missing answer" in str(exc)
    else:
        raise AssertionError("expected KevError")


def test_kev_rejects_non_http():
    try:
        KevBackend(base_url="file:///etc/passwd")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for file:// URL")


def test_kev_media_dropped_and_reported(monkeypatch):
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda req, timeout=None: _FakeResp(_kev_payload()))
    out = KevBackend(base_url="http://kev-test:8008").systemone(
        "s", _questions(), images=["http://img/1.png"],
        videos=["http://vid/1.mp4"])
    assert out["_meta"]["media_dropped"] == {"images": 1, "videos": 1}


def test_kev_env_config(monkeypatch):
    monkeypatch.setenv("KEV_BASE_URL", "http://gpu-box:8008/")
    monkeypatch.setenv("KEV_MODEL", "jaredpalmer/kev-4b")
    b = KevBackend()
    assert b.base_url == "http://gpu-box:8008"
    assert b.model_name == "jaredpalmer/kev-4b"


def test_kev_engine_wiring():
    assert engine_backend_name(KevBackend.__new__(KevBackend)) == "kev"
    assert "kev" in shim_module.ENGINE_CHOICES


def test_kev_registry_pack_has_four_routable_tiers():
    import systemone.shim as shim

    pack = json.load(open(os.path.join(
        os.path.dirname(shim.__file__), "kev_registry.json")))
    cands = shim.candidates_from_registry(pack)
    assert [c["tier"] for c in cands] == [
        "kev-0.8b", "kev-4b", "kev-9b", "kev-27b"]
    assert all(c["model_id"].startswith("jaredpalmer/kev-") for c in cands)
