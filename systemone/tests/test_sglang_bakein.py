"""Tests for the baked-in SGLang engine: selection, hybrid, client, health.

No torch, no network (numpy is a base dependency). These tests prove the
SGLang path works on a slim install.
"""

import importlib.util
import json
import os
import sys
import threading
import urllib.request

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import systemone  # noqa: E402
from systemone import (  # noqa: E402
    CascadeBackend,
    HybridBackend,
    SGLangBackend,
    SGLangError,
)
from systemone import shim as shim_module  # noqa: E402
from systemone.client import SystemOneClient  # noqa: E402
from systemone.shim import (  # noqa: E402
    create_engine,
    engine_backend_name,
    serve,
)


class StubEngine:
    model_name = "stub"
    backend_name = "stub"

    def __init__(self, confidence=0.9):
        self.confidence = confidence
        self.calls = 0

    def systemone(self, state, questions, **kwargs):
        self.calls += 1
        answers = {}
        for q in questions:
            if q["type"] == "choice":
                opts = q["options"]
                n = len(opts)
                answers[q["name"]] = {
                    "type": "choice", "choice": opts[0],
                    "probabilities": {o: 1.0 / n for o in opts},
                    "confidence": self.confidence,
                }
            elif q["type"] == "score":
                lv = q["levels"]
                n = len(lv)
                answers[q["name"]] = {
                    "type": "score", "level": lv[0],
                    "distribution": {x: 1.0 / n for x in lv},
                    "confidence": self.confidence,
                }
            else:
                answers[q["name"]] = {
                    "type": "noul", "probability": 0.8, "answer": True,
                    "confidence": self.confidence,
                }
        answers["_meta"] = {"model": "stub", "latency_ms": 1.0}
        return answers


class FailingSGLang:
    model_name = "sglang-dead"
    backend_name = "sglang"

    def systemone(self, state, questions, images=None):
        raise SGLangError("boom", hint="test failure")


def _questions():
    return [
        {"name": "action", "type": "choice",
         "options": ["up", "down"], "prompt": "move?"},
        {"name": "stuck", "type": "noul", "statement": "stuck?"},
    ]


# -- slim importability -------------------------------------------------


def test_slim_imports_pull_no_torch():
    if importlib.util.find_spec("torch") is not None:
        return  # heavy env: nothing to prove
    assert "torch" not in sys.modules
    assert "gliclass" not in sys.modules
    assert systemone.SGLangBackend is SGLangBackend
    assert systemone.HybridBackend is HybridBackend
    assert callable(systemone.serve_shim)


def test_heavy_names_raise_helpful_error_without_torch():
    if importlib.util.find_spec("torch") is not None:
        return
    try:
        systemone.SystemOne
    except ImportError as exc:
        assert "systemone[local]" in str(exc)
    else:
        raise AssertionError("expected ImportError for SystemOne on slim install")


def test_unknown_lazy_name_raises_attribute_error():
    try:
        systemone.no_such_name_at_all
    except AttributeError:
        pass
    else:
        raise AssertionError("expected AttributeError")


# -- patterns canonical-source parity ---------------------------------------


def test_patterns_are_canonical_light_source():
    import systemone.patterns as patterns

    assert shim_module.MAX_STATE_CHARS is patterns.MAX_STATE_CHARS == 6000
    assert shim_module.validate_choice is patterns.validate_choice
    assert systemone.SystemOneError is patterns.SystemOneError
    assert systemone.make_questions is patterns.make_questions
    assert systemone.StallGuard is patterns.StallGuard


def test_api_reexports_patterns_when_heavy():
    if importlib.util.find_spec("torch") is None:
        return  # slim env: nothing to compare against
    import systemone.api as api
    import systemone.patterns as patterns

    assert api.MAX_STATE_CHARS is patterns.MAX_STATE_CHARS
    assert api.SystemOneError is patterns.SystemOneError
    assert api.validate_choice is patterns.validate_choice
    assert api.make_questions is patterns.make_questions
    assert shim_module.SystemOne is api.SystemOne


# -- engine selection ---------------------------------------------------


def test_create_engine_rejects_unknown_name():
    try:
        create_engine("bogus-engine")
    except ValueError as exc:
        assert "SYSTEMONE_ENGINE" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_create_engine_sglang_when_healthy(monkeypatch):
    monkeypatch.setattr(
        SGLangBackend, "health", lambda self: True
    )
    eng = create_engine("sglang")
    assert isinstance(eng, SGLangBackend)
    assert engine_backend_name(eng) == "sglang"


def test_create_engine_sglang_fails_open_to_local(monkeypatch):
    monkeypatch.setattr(SGLangBackend, "health", lambda self: False)
    fake = StubEngine()
    monkeypatch.setattr(shim_module, "SystemOne", lambda model_name=None: fake)
    eng = create_engine("sglang")
    assert eng is fake


def test_create_engine_sglang_unreachable_slim_raises(monkeypatch):
    monkeypatch.setattr(SGLangBackend, "health", lambda self: False)
    monkeypatch.setattr(shim_module, "SystemOne", None)
    try:
        create_engine("sglang")
    except SGLangError as exc:
        assert "no local engine" in str(exc)
    else:
        raise AssertionError("expected SGLangError")


def test_create_engine_local_slim_raises_helpful(monkeypatch):
    monkeypatch.setattr(shim_module, "SystemOne", None)
    try:
        create_engine("local")
    except ImportError as exc:
        assert "systemone[local]" in str(exc)
    else:
        raise AssertionError("expected ImportError")


def test_auto_uses_sglang_only_when_configured(monkeypatch):
    monkeypatch.setattr(SGLangBackend, "health", lambda self: True)
    fake = StubEngine()
    monkeypatch.setattr(shim_module, "SystemOne", lambda model_name=None: fake)
    # No SGLANG_BASE_URL -> local, no probe side effects observed.
    monkeypatch.delenv("SGLANG_BASE_URL", raising=False)
    monkeypatch.delenv("SYSTEMONE_ENGINE", raising=False)
    assert create_engine("auto") is fake
    # Explicit SGLANG_BASE_URL + healthy -> SGLang.
    monkeypatch.setenv("SGLANG_BASE_URL", "http://gpu-box:30000")
    eng = create_engine("auto")
    assert isinstance(eng, SGLangBackend)


def test_auto_falls_back_when_sglang_down(monkeypatch):
    monkeypatch.setattr(SGLangBackend, "health", lambda self: False)
    fake = StubEngine()
    monkeypatch.setattr(shim_module, "SystemOne", lambda model_name=None: fake)
    monkeypatch.setenv("SGLANG_BASE_URL", "http://gpu-box:30000")
    assert create_engine("auto") is fake


def test_engine_name_env_var_honored(monkeypatch):
    monkeypatch.setattr(SGLangBackend, "health", lambda self: True)
    monkeypatch.setenv("SYSTEMONE_ENGINE", "sglang")
    assert isinstance(create_engine(None), SGLangBackend)


def test_engine_backend_name_labels():
    assert engine_backend_name(SGLangBackend()) == "sglang"
    hybrid = HybridBackend(local_engine=StubEngine(), sglang=None)
    assert engine_backend_name(hybrid) == "hybrid"
    assert engine_backend_name(StubEngine()) == "custom"


# -- hybrid escalation --------------------------------------------------


def test_hybrid_stays_local_when_confident():
    local = StubEngine(confidence=0.95)
    sglang = StubEngine(confidence=0.99)
    hybrid = HybridBackend(local_engine=local, sglang=sglang, escalate_below=0.6)
    out = hybrid.systemone("state", _questions())
    assert local.calls == 1
    assert sglang.calls == 0
    assert out["_meta"]["backend"] == "hybrid/local"
    assert out["_meta"]["escalated"] is False


def test_hybrid_escalates_when_unsure():
    local = StubEngine(confidence=0.2)
    sglang = StubEngine(confidence=0.99)
    hybrid = HybridBackend(local_engine=local, sglang=sglang, escalate_below=0.6)
    out = hybrid.systemone("state", _questions())
    assert local.calls == 1
    assert sglang.calls == 1
    assert out["_meta"]["backend"] == "hybrid/sglang"
    assert out["_meta"]["escalated"] is True
    assert out["_meta"]["min_local_confidence"] == 0.2


def test_hybrid_escalation_failure_fails_open():
    local = StubEngine(confidence=0.2)
    hybrid = HybridBackend(
        local_engine=local, sglang=FailingSGLang(), escalate_below=0.6
    )
    out = hybrid.systemone("state", _questions())
    assert out["_meta"]["backend"] == "hybrid/local"
    assert "boom" in out["_meta"]["escalation_error"]
    assert out["action"]["choice"] == "up"  # local answers preserved


def test_cascade_stays_cheap_when_confident():
    cheap = StubEngine(confidence=0.95)
    pricey = StubEngine(confidence=0.99)
    cascade = CascadeBackend([cheap, pricey], escalate_below=0.6,
                             costs=[0.1, 1.0])
    out = cascade.systemone("state", _questions())
    assert cheap.calls == 1 and pricey.calls == 0
    assert out["_meta"]["backend"] == "cascade"
    assert out["_meta"]["stage"] == 0
    assert out["_meta"]["escalated"] is False
    assert out["_meta"]["cost_spent"] == pytest.approx(0.1)


def test_cascade_escalates_through_stages():
    weak = StubEngine(confidence=0.2)
    mid = StubEngine(confidence=0.5)
    strong = StubEngine(confidence=0.99)
    cascade = CascadeBackend([weak, mid, strong], escalate_below=0.6,
                             costs=[0.1, 0.3, 1.0])
    out = cascade.systemone("state", _questions())
    assert (weak.calls, mid.calls, strong.calls) == (1, 1, 1)
    assert out["_meta"]["stage"] == 2
    assert out["_meta"]["stages_run"] == 3
    assert out["_meta"]["cost_spent"] == pytest.approx(1.4)
    assert out["_meta"]["min_confidence"] == pytest.approx(0.99)


def test_cascade_budget_stops_escalation():
    weak = StubEngine(confidence=0.2)
    strong = StubEngine(confidence=0.99)
    cascade = CascadeBackend([weak, strong], escalate_below=0.6,
                             budget=0.5, costs=[0.1, 1.0])
    out = cascade.systemone("state", _questions())
    assert strong.calls == 0  # 0.1 + 1.0 exceeds the 0.5 budget
    assert out["_meta"]["budget_exhausted"] is True
    assert out["_meta"]["min_confidence"] == pytest.approx(0.2)
    # ...but a fitting budget escalates
    strong.calls = 0
    cascade = CascadeBackend([weak, strong], escalate_below=0.6,
                             budget=1.5, costs=[0.1, 1.0])
    out = cascade.systemone("state", _questions())
    assert strong.calls == 1
    assert out["_meta"]["budget_exhausted"] is False


def test_cascade_stage_failure_fails_open():
    weak = StubEngine(confidence=0.2)
    cascade = CascadeBackend([weak, FailingSGLang()], escalate_below=0.6)
    out = cascade.systemone("state", _questions())
    assert out["_meta"]["stage"] == 0
    assert "boom" in out["_meta"]["escalation_error"]
    assert out["action"]["choice"] == "up"


def test_cascade_validation():
    with pytest.raises(ValueError):
        CascadeBackend([])
    with pytest.raises(ValueError):
        CascadeBackend([StubEngine()], escalate_below=1.5)
    with pytest.raises(ValueError):
        CascadeBackend([StubEngine(), StubEngine()], costs=[0.1])
    with pytest.raises(ValueError):
        CascadeBackend([StubEngine()], costs=[-1.0])
    with pytest.raises(ValueError):
        CascadeBackend([StubEngine()], budget=-1.0)


def test_hybrid_non_sglang_escalation_error_fails_open():
    class WeirdSGLang:
        def systemone(self, state, questions, **kwargs):
            raise ValueError("weird backend bug")

    local = StubEngine(confidence=0.2)
    hybrid = HybridBackend(
        local_engine=local, sglang=WeirdSGLang(), escalate_below=0.6
    )
    out = hybrid.systemone("state", _questions())
    assert out["_meta"]["backend"] == "hybrid/local"
    assert "weird backend bug" in out["_meta"]["escalation_error"]
    assert out["action"]["choice"] == "up"  # local answers preserved


def test_hybrid_without_sglang_is_pure_local():
    local = StubEngine(confidence=0.1)
    hybrid = HybridBackend(local_engine=local, sglang=None, escalate_below=0.6)
    out = hybrid.systemone("state", _questions())
    assert out["_meta"]["backend"] == "hybrid/local"
    assert out["_meta"]["escalated"] is False


def test_hybrid_without_local_delegates_to_sglang():
    sglang = StubEngine(confidence=0.9)
    hybrid = HybridBackend(local_engine=None, sglang=sglang)
    out = hybrid.systemone("state", _questions())
    assert out["_meta"]["backend"] == "hybrid/sglang"
    assert out["_meta"]["escalated"] is True


def test_hybrid_with_neither_side_raises():
    hybrid = HybridBackend(local_engine=None, sglang=None)
    try:
        hybrid.systemone("state", _questions())
    except SGLangError:
        pass
    else:
        raise AssertionError("expected SGLangError")


# -- loop adapter -------------------------------------------------------------


def test_decide_fn_for_matches_loop_protocol():
    from systemone import decide_fn_for

    seen = {}

    class MediaEngine:
        def systemone(self, state, questions, images=None, videos=None):
            seen["kwargs"] = (images, videos)
            return {"a": {"type": "noul"}, "_meta": {}}

    fn = decide_fn_for(MediaEngine())
    fn("s", [], ["i"], ["v"])
    assert seen["kwargs"] == (["i"], ["v"])
    fn("s", [], [], [])
    assert seen["kwargs"] == (None, None)

    class OldStub:
        def systemone(self, state, questions):
            return {"a": 1}

    assert decide_fn_for(OldStub())("s", [], ["i"], ["v"]) == {"a": 1}


# -- confidence uniformity ------------------------------------------------


def test_backend_confidence_uses_typesafe_helpers(monkeypatch):
    from systemone.patterns import choice_confidence, score_confidence

    def fake_urlopen(req, timeout=None):
        return _FakeResp({"answers": {
            "c": {"type": "choice", "choice": "up",
                  "probabilities": {"up": 0.7, "down": 0.3}},
            "s": {"type": "score",
                  "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}},
        }})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    out = SGLangBackend().systemone("s", [
        {"name": "c", "type": "choice", "options": ["up", "down"]},
        {"name": "s", "type": "score", "levels": ["0", "1", "2"]},
    ])
    assert out["c"]["confidence"] == choice_confidence([0.7, 0.3])
    assert out["s"]["confidence"] == score_confidence([0.1, 0.2, 0.7])


# -- client + shim HTTP -------------------------------------------------


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_client_decisions_posts_sglang_dialect(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data.decode())
        return _FakeResp({"answers": {"a": {"type": "yes_no", "answer": True}}})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = SystemOneClient("http://shim-test:8765")
    out = client.decisions("state", [
        {"id": "a", "type": "yes_no", "question": "ok?"},
    ])
    assert seen["url"] == "http://shim-test:8765/v1/decisions"
    assert seen["body"]["input"] == "state"
    assert out["answers"]["a"]["answer"] is True


def test_client_systemone_posts_typesafe_dialect(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data.decode())
        return _FakeResp({"answers": {}})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    out = SystemOneClient("http://shim-test:8765").systemone(
        "state", {"q": {"type": "choice", "criteria": {"a": "A"}}}
    )
    assert seen["url"] == "http://shim-test:8765/v1/systemone"
    assert seen["body"]["state"] == "state"
    assert out == {"answers": {}}


def _serving(server):
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return t


def test_healthz_reports_backend():
    server = serve(0, engine=StubEngine())
    port = server.server_address[1]
    assert server.engine_backend == "custom"
    t = _serving(server)
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/healthz", timeout=10
        ) as resp:
            body = json.loads(resp.read())
        assert body["ok"] is True
        assert body["backend"] == "custom"
    finally:
        server.shutdown()
        t.join(timeout=5)


def test_healthz_reports_sglang_backend():
    server = serve(0, engine=SGLangBackend())
    port = server.server_address[1]
    assert server.engine_backend == "sglang"
    t = _serving(server)
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/healthz", timeout=10
        ) as resp:
            body = json.loads(resp.read())
        assert body["backend"] == "sglang"
    finally:
        server.shutdown()
        t.join(timeout=5)


def test_serve_accepts_engine_name(monkeypatch):
    monkeypatch.setattr(SGLangBackend, "health", lambda self: True)
    server = serve(0, engine_name="sglang")
    try:
        assert isinstance(server.engine, SGLangBackend)
        assert server.engine_backend == "sglang"
    finally:
        server.server_close()


# -- URL scheme guards ----------------------------------------------------


def test_client_rejects_non_http_base_url():
    try:
        SystemOneClient("file:///etc/passwd")
    except ValueError as exc:
        assert "http(s)" in str(exc)
    else:
        raise AssertionError("expected ValueError for file:// shim URL")


def test_sglang_backend_rejects_non_http_base_url():
    try:
        SGLangBackend(base_url="ftp://example.com/x")
    except ValueError as exc:
        assert "http(s)" in str(exc)
    else:
        raise AssertionError("expected ValueError for ftp:// SGLang URL")


def test_jeff1_url_rejects_non_http(monkeypatch):
    from systemone import jeff1

    monkeypatch.setenv("SYSTEMONE_JEFF1_URL", "file:///etc/passwd")
    try:
        jeff1.jeff1_url()
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for file:// sidecar URL")
    # ...and the request path fails open (None) instead of raising.
    monkeypatch.setenv("SYSTEMONE_JEFF1", "1")
    assert jeff1._post("/x", {}, timeout=1) is None


def test_lmstudio_teacher_rejects_non_http():
    from systemone.distill import LMStudioTeacher

    try:
        LMStudioTeacher(base_url="file:///etc/passwd")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for file:// teacher URL")


def test_fetch_lmstudio_models_rejects_non_http(monkeypatch):
    from systemone import scoring

    monkeypatch.setenv("LMSTUDIO_BASE_URL", "file:///etc/passwd")
    assert scoring.fetch_lmstudio_models() is None  # fail-open, no request
