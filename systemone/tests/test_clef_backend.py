"""Tests for the Clef backend (Cloudflare clef / clef-flash, run locally).

Weight loading is stubbed — tests never download weights or import torch.
Media decoding uses real PIL data URLs / local files; the joint-schema
forward pass is a canned Clef-shaped ``systemone()`` response.
"""

import base64
import io
import os
import sys
import urllib.request

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from systemone import ClefBackend, ClefError  # noqa: E402
from systemone import shim as shim_module  # noqa: E402
from systemone.shim import (  # noqa: E402
    ENGINE_CHOICES,
    create_engine,
    engine_backend_name,
)


def _png_bytes():
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (2, 2), (255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def _data_url():
    return "data:image/png;base64," + base64.b64encode(_png_bytes()).decode()


class _StubCSM:
    """Stands in for the release's joint_schema_model module."""

    def __init__(self, answers):
        self.answers = answers
        self.seen = None

    def systemone(self, model, processor, request, max_length=16384):
        self.seen = dict(request)
        self.seen_max_length = max_length
        return {
            "model": request["model"],
            "answers": self.answers,
            "usage": {"input_tokens": 42, "output_tokens": 0},
        }


def _backend(monkeypatch, answers, **kwargs):
    stub = _StubCSM(answers)

    def fake_load(self):
        self._model = object()
        self._processor = object()
        self._csm = stub

    monkeypatch.setattr(ClefBackend, "_load", fake_load)
    backend = ClefBackend(**kwargs)
    backend._stub = stub
    return backend


def _questions():
    return [
        {"name": "dept", "type": "choice",
         "options": [{"name": "billing"}, {"name": "technical"}],
         "prompt": "Which team?"},
        {"name": "urgency", "type": "score",
         "levels": ["Can wait", "Today"]},
        {"name": "outage", "type": "noul",
         "statement": "Is a service down?"},
    ]


def _answers():
    return {
        # Clef-shaped: choice keyed by name, score by index + legend.
        "dept": {"type": "choice", "choice": "billing",
                 "probabilities": {"billing": 0.7, "technical": 0.3}},
        "urgency": {"type": "score",
                    "probabilities": {"0": 0.2, "1": 0.8},
                    "legend": {"0": "Can wait", "1": "Today"}},
        "outage": {"type": "noul", "noul": 0.9},
    }


def test_maps_all_types_with_reference_confidence(monkeypatch):
    from systemone.patterns import (
        choice_confidence,
        noul_confidence,
        score_confidence,
    )

    backend = _backend(monkeypatch, _answers())
    out = backend.systemone("checkout is down", _questions())
    assert out["dept"]["choice"] == "billing"
    assert out["dept"]["probabilities"] == {"billing": 0.7, "technical": 0.3}
    assert out["dept"]["confidence"] == pytest.approx(
        choice_confidence([0.7, 0.3]))
    assert out["urgency"]["level"] == "Today"
    assert out["urgency"]["distribution"] == {"Can wait": 0.2, "Today": 0.8}
    assert out["urgency"]["score"] == pytest.approx(0.8)
    assert out["urgency"]["confidence"] == pytest.approx(
        score_confidence([0.2, 0.8]))
    assert out["urgency"]["legend"] == {"Can wait": "Can wait", "Today": "Today"}
    assert out["outage"] == {"type": "noul", "probability": 0.9,
                             "answer": True,
                             "confidence": noul_confidence(0.9)}
    meta = out["_meta"]
    assert meta["backend"] == "clef"
    assert meta["model"] == "Cloudflare/clef-flash"
    assert meta["usage"] == {"input_tokens": 42, "output_tokens": 0}
    assert meta["input_tokens"] == 42
    assert "media_dropped" not in meta


def test_request_shape_matches_clef_record(monkeypatch):
    backend = _backend(monkeypatch, _answers())
    backend.systemone("checkout is down", _questions())
    seen = backend._stub.seen
    assert seen["model"] == "Cloudflare/clef-flash"
    assert seen["state"] == "checkout is down"
    assert seen["questions"]["dept"]["criteria"] == {
        "billing": None, "technical": None}
    assert seen["questions"]["urgency"]["criteria"] == ["Can wait", "Today"]
    assert seen["questions"]["outage"] == {
        "type": "noul", "instructions": "Is a service down?"}
    assert "images" not in seen and "videos" not in seen
    assert backend._stub.seen_max_length == 16384


def test_score_name_keyed_probs_accepted(monkeypatch):
    answers = {"urgency": {"type": "score",
                           "probabilities": {"Can wait": 0.25, "Today": 0.75}}}
    backend = _backend(monkeypatch, answers)
    out = backend.systemone("s", [
        {"name": "urgency", "type": "score", "levels": ["Can wait", "Today"]}])
    assert out["urgency"]["distribution"] == {"Can wait": 0.25, "Today": 0.75}


def test_bad_answers_raise_loud(monkeypatch):
    backend = _backend(monkeypatch, {})
    with pytest.raises(ClefError):
        backend.systemone("s", _questions()[:1])
    backend = _backend(monkeypatch, {"dept": {"type": "choice"}})
    with pytest.raises(ClefError):
        backend.systemone("s", _questions()[:1])
    backend = _backend(monkeypatch, {"outage": {"type": "noul", "noul": "x"}})
    with pytest.raises(ClefError):
        backend.systemone("s", _questions()[2:3])
    with pytest.raises(ClefError):
        backend.systemone("s", [])
    with pytest.raises(ClefError):
        backend.systemone("s", [{"name": "x", "type": "mystery"}])
    with pytest.raises(ClefError):
        backend.systemone("s", [{"name": "x", "type": "choice",
                                 "options": ["only"]}])


def test_images_decode_and_reach_the_forward_pass(monkeypatch):
    backend = _backend(monkeypatch, _answers())
    out = backend.systemone("review the receipt", _questions(),
                            images=[_data_url()])
    seen = backend._stub.seen
    assert len(seen["images"]) == 1
    assert seen["images"][0].size == (2, 2)
    assert out["_meta"].get("media_dropped") is None


def test_image_file_path_and_pil_passthrough(monkeypatch, tmp_path):
    from PIL import Image

    backend = _backend(monkeypatch, _answers())
    path = tmp_path / "shot.png"
    path.write_bytes(_png_bytes())
    out = backend.systemone("s", _questions(),
                            images=[str(path), Image.new("RGB", (1, 1))])
    assert len(backend._stub.seen["images"]) == 2
    assert out["_meta"].get("media_dropped") is None


def test_image_url_download(monkeypatch):
    blob = _png_bytes()

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n=-1):
            return blob[:n] if n is not None and n >= 0 else blob

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout: _Resp())
    backend = _backend(monkeypatch, _answers())
    backend.systemone("s", _questions(),
                      images=["https://example.com/shot.png"])
    assert len(backend._stub.seen["images"]) == 1


def test_undecodable_media_drops_fail_open_with_reasons(monkeypatch):
    backend = _backend(monkeypatch, _answers())
    out = backend.systemone(
        "s", _questions(),
        images=["not-a-path.png", "data:image/png;base64,!!!"],
        videos=["not-frames"])
    dropped = out["_meta"]["media_dropped"]
    assert dropped["images"] == 2
    assert dropped["videos"] == 1
    assert len(dropped["reasons"]) == 3
    # The request still judged on the text state.
    assert out["dept"]["choice"] == "billing"


def test_video_frame_arrays_pass_through(monkeypatch):
    np = pytest.importorskip("numpy")
    backend = _backend(monkeypatch, _answers())
    frames = [np.zeros((2, 2, 3), dtype="uint8")]
    backend.systemone("s", _questions(), videos=frames)
    assert backend._stub.seen["videos"] == frames


def test_image_size_cap(monkeypatch):
    backend = _backend(monkeypatch, _answers(), image_max_mb=0.000001)
    out = backend.systemone("s", _questions(), images=[_data_url()])
    assert out["_meta"]["media_dropped"]["images"] == 1


def test_env_config_and_validation(monkeypatch):
    monkeypatch.setenv("CLEF_MODEL_ID", "Cloudflare/clef")
    monkeypatch.setenv("CLEF_DTYPE", "float32")
    monkeypatch.setenv("CLEF_MAX_LENGTH", "4096")
    backend = _backend(monkeypatch, _answers())
    assert backend.model_id == "Cloudflare/clef"
    assert backend.dtype_name == "float32"
    assert backend.max_length == 4096
    backend.systemone("s", _questions()[:1])
    assert backend._stub.seen["model"] == "Cloudflare/clef"
    with pytest.raises(ClefError):
        _backend(monkeypatch, _answers(), dtype="int8")
    with pytest.raises(ClefError):
        _backend(monkeypatch, _answers(), max_length=0)
    with pytest.raises(ClefError):
        _backend(monkeypatch, _answers(), image_max_mb=0)


def test_health_true_once_loaded(monkeypatch):
    assert _backend(monkeypatch, _answers()).health() is True


def test_missing_torch_raises_helpful_clef_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ClefError, match="systemone\\[clef\\]"):
        ClefBackend()


def test_engine_registration(monkeypatch):
    assert "clef" in ENGINE_CHOICES

    made = {}

    class _Fake:
        backend_name = "clef"

        def __init__(self):
            made["yes"] = True

    # A real backend instance reports the clef label (check before the
    # patch below swaps the shim's reference for a fake).
    real = ClefBackend.__new__(ClefBackend)
    assert engine_backend_name(real) == "clef"
    monkeypatch.setattr(shim_module, "ClefBackend", _Fake)
    assert isinstance(create_engine("clef"), _Fake)
    assert made == {"yes": True}


def test_create_engine_clef_fails_open_to_local(monkeypatch):
    import systemone.shim as shim

    def _boom():
        raise ClefError("no weights here")

    class _Local:
        pass

    monkeypatch.setattr(shim_module, "ClefBackend", _boom)
    monkeypatch.setattr(shim, "SystemOne", _Local)
    monkeypatch.setenv("SYSTEMONE_MODEL", "x")
    # _local() constructs SystemOne(model_name=...); accept any kwargs.
    _Local.__init__ = lambda self, **kw: None
    assert isinstance(create_engine("clef"), _Local)
