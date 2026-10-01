"""Device auto-detection: CUDA > Apple MPS > CPU. No model needed."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

torch = pytest.importorskip("torch", reason="needs systemone[local]")

import systemone.api as api
from systemone.api import SystemOne, default_device


def test_default_device_prefers_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert default_device() == "cuda"


def test_default_device_mps_when_no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert default_device() == "mps"


def test_default_device_cpu_last_resort(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert default_device() == "cpu"


def _stub_loading(monkeypatch):
    """Stub model/tokenizer/pipeline so SystemOne() never touches disk/net."""

    class _Dummy:
        pass

    monkeypatch.setattr(
        api.GLiClassModel,
        "from_pretrained",
        classmethod(lambda cls, _id: _Dummy()),
    )
    monkeypatch.setattr(
        api.AutoTokenizer,
        "from_pretrained",
        classmethod(lambda cls, _id: _Dummy()),
    )
    monkeypatch.setattr(
        api,
        "ZeroShotClassificationPipeline",
        lambda model, tokenizer, device=None, **kw: None,
    )


@pytest.mark.parametrize("device_arg", [None, "auto", "AUTO", " Auto "])
def test_auto_device_string_resolves(monkeypatch, device_arg):
    _stub_loading(monkeypatch)
    eng = SystemOne(model_name="dummy-checkpoint", device=device_arg)
    assert eng.device == default_device()


def test_explicit_device_is_honored(monkeypatch):
    _stub_loading(monkeypatch)
    eng = SystemOne(model_name="dummy-checkpoint", device="cpu")
    assert eng.device == "cpu"


def test_revision_pin_reaches_from_pretrained(monkeypatch):
    seen = {}

    class _Dummy:
        pass

    def _capture(kind):
        def go(cls, model_id, **kwargs):
            seen[kind] = (model_id, kwargs)
            return _Dummy()
        return classmethod(go)

    monkeypatch.setattr(api.GLiClassModel, "from_pretrained", _capture("model"))
    monkeypatch.setattr(api.AutoTokenizer, "from_pretrained", _capture("tok"))
    monkeypatch.setattr(
        api, "ZeroShotClassificationPipeline",
        lambda model, tokenizer, device=None, **kw: None,
    )
    eng = SystemOne(model_name="dummy-checkpoint", device="cpu",
                    revision="abc123")
    assert eng.revision == "abc123"
    assert seen["model"][1] == {"revision": "abc123"}
    assert seen["tok"][1] == {"revision": "abc123"}


def test_revision_defaults_unpinned(monkeypatch):
    monkeypatch.delenv("SYSTEMONE_REVISION", raising=False)
    _stub_loading(monkeypatch)
    eng = SystemOne(model_name="dummy-checkpoint", device="cpu")
    assert eng.revision is None
