"""Tests for Clef (Cloudflare/clef) compatibility on /v1/systemone.

Covers the pulled features: top-level images/videos/media_kwargs, score
legends, the "noul" P(true) alias, usage.output_tokens=0, and optional
instructions with question-ID fallback. No model needed.
"""

import json
import os
import sys
import threading
import urllib.error
import urllib.request

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from systemone.rerank_backend import RerankBackend  # noqa: E402
from systemone.shim import (  # noqa: E402
    _engine_supports,
    serve,
    translate_answers,
    translate_media,
    translate_question,
)


class StubNoMedia:
    model_name = "stub"

    def systemone(self, state, questions):
        return _stub_answers()


class StubMedia:
    """Media-capable stub: consumes images, drops videos."""

    model_name = "stub-media"
    seen = None

    def systemone(self, state, questions, images=None, videos=None):
        StubMedia.seen = (list(images or []), list(videos or []))
        out = _stub_answers()
        if videos:
            out["_meta"]["media_dropped"] = {
                "images": 0, "videos": len(videos)}
        return out


def _stub_answers():
    return {
        "urgency": {
            "type": "score", "level": "Today",
            "distribution": {"Can wait": 0.2, "Today": 0.8},
            "confidence": 0.8,
            "legend": {"Can wait": "Can wait", "Today": "Today"},
        },
        "outage": {
            "type": "noul", "probability": 0.9, "answer": True,
            "confidence": 0.9,
        },
        "_meta": {"model": "stub", "latency_ms": 1.0},
    }


def _clef_body():
    return {
        "model": "clef",
        "state": "Our checkout started returning errors.",
        "questions": {
            "urgency": {"type": "score",
                        "criteria": ["Can wait", "Today"]},
            "outage": {"type": "noul",
                       "criteria": {"true": "A service is down.",
                                    "false": "All services are up."}},
        },
    }


# -- translators --------------------------------------------------------


def test_score_question_carries_legend():
    q = translate_question("urgency", {"type": "score",
                                       "criteria": ["Can wait", "Today"]})
    assert q["levels"] == ["Can wait", "Today"]
    assert q["legend"] == {"Can wait": "Can wait", "Today": "Today"}
    q = translate_question("u", {"type": "score",
                                 "criteria": {"low": "Minor.", "high": "Bad."}})
    assert q["legend"] == {"low": "Minor.", "high": "Bad."}


def test_missing_instructions_falls_back_to_id():
    q = translate_question("urgency", {"type": "score",
                                       "criteria": ["a", "b"]})
    assert "urgency" in q["prompt"]
    assert q["prompt"].startswith("Question 'urgency'.")
    q = translate_question("x", {"type": "yes_no"})
    assert q["statement"] == "x"


def test_unknown_qtype_and_empty_options_rejected():
    with pytest.raises(ValueError, match="unknown type"):
        translate_question("x", {"type": "bogus", "criteria": {"a": "A"}})
    with pytest.raises(ValueError, match=">= 2 options"):
        translate_question("x", {"type": "choice", "criteria": {}})
    with pytest.raises(ValueError, match=">= 2 options"):
        translate_question("x", {"type": "score", "criteria": []})


def test_noul_criteria_descriptions_reach_statement():
    q = translate_question("outage", {"type": "noul",
                                      "criteria": {"true": "Down.",
                                                   "false": "Up."}})
    assert "Down." in q["statement"] and "Up." in q["statement"]


def test_translate_media_accepts_clef_forms():
    images, videos = translate_media({
        "images": ["https://e.com/a.png", {"image": "data:xyz"}],
        "videos": [["frame1"], "https://e.com/v.mp4"],
        "media_kwargs": {"fps": 2},
    })
    assert len(images) == 2 and len(videos) == 2
    assert translate_media({}) == ([], [])


def test_translate_media_rejects_bad_shapes():
    with pytest.raises(ValueError):
        translate_media({"images": "nope"})
    with pytest.raises(ValueError):
        translate_media({"videos": [{"nope": 1}]})
    with pytest.raises(ValueError):
        translate_media({"images": [42]})
    with pytest.raises(ValueError):
        translate_media({"media_kwargs": ["nope"]})


def test_engine_supports_probe():
    assert _engine_supports(StubMedia(), "images") is True
    assert _engine_supports(StubMedia(), "videos") is True
    assert _engine_supports(StubNoMedia(), "images") is False
    assert _engine_supports(StubNoMedia(), "videos") is False


def test_translate_answers_legend_and_noul_alias():
    out = translate_answers(_stub_answers())
    assert out["urgency"]["legend"] == {"Can wait": "Can wait",
                                        "Today": "Today"}
    assert out["outage"]["noul"] == 0.9
    assert out["outage"]["probability"] == 0.9


def test_translate_answers_legend_identity_fallback():
    out = translate_answers({"s": {
        "type": "score", "level": "b",
        "distribution": {"a": 0.3, "b": 0.7}, "confidence": 0.7}})
    assert out["s"]["legend"] == {"a": "a", "b": "b"}


# -- HTTP ---------------------------------------------------------------


def _post(url, body):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
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


def test_serve_clef_body_roundtrip():
    server, t = _serve(StubNoMedia())
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        status, payload = _post(f"{base}/v1/systemone", _clef_body())
        assert status == 200
        assert payload["usage"] == {"output_tokens": 0}
        assert payload["answers"]["urgency"]["legend"] == {
            "Can wait": "Can wait", "Today": "Today"}
        assert payload["answers"]["outage"]["noul"] == 0.9
        assert "media" not in payload and "warnings" not in payload
    finally:
        _close(server, t)


def test_serve_media_dropped_warns_on_text_stub():
    server, t = _serve(StubNoMedia())
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        body = _clef_body()
        body["images"] = ["https://e.com/a.png"]
        body["videos"] = ["https://e.com/v.mp4"]
        status, payload = _post(f"{base}/v1/systemone", body)
        assert status == 200
        assert payload["media"] == {"images": 1, "videos": 1}
        assert len(payload["warnings"]) == 1
        assert "image" in payload["warnings"][0]
    finally:
        _close(server, t)


def test_serve_media_forwarded_to_capable_engine():
    StubMedia.seen = None
    server, t = _serve(StubMedia())
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        body = _clef_body()
        body["images"] = ["https://e.com/a.png"]
        body["videos"] = ["https://e.com/v.mp4"]
        status, payload = _post(f"{base}/v1/systemone", body)
        assert status == 200
        assert StubMedia.seen == (["https://e.com/a.png"],
                                  ["https://e.com/v.mp4"])
        assert payload["media"] == {"images": 1, "videos": 1}
        # Images consumed; only the dropped video warns.
        assert payload["warnings"] == [
            "1 video(s) dropped: the custom engine has no media path for them"]
    finally:
        _close(server, t)


def test_serve_bad_media_is_400():
    server, t = _serve(StubNoMedia())
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        body = _clef_body()
        body["images"] = "nope"
        status, _ = _post(f"{base}/v1/systemone", body)
        assert status == 400
    finally:
        _close(server, t)


def test_usage_carries_input_tokens_when_reported():
    class Counting(StubNoMedia):
        def systemone(self, state, questions):
            out = _stub_answers()
            out["_meta"]["input_tokens"] = 128
            return out

    server, t = _serve(Counting())
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        status, payload = _post(f"{base}/v1/systemone", _clef_body())
        assert payload["usage"] == {"output_tokens": 0, "input_tokens": 128}
    finally:
        _close(server, t)


# -- engines ------------------------------------------------------------


def test_jevk5_forwards_media_in_body(monkeypatch):
    from systemone.jevk5_backend import JevK5ServerBackend

    seen = {}

    class FakeResp:
        status = 200

        def read(self):
            return json.dumps({"answers": {
                "ok": {"type": "noul", "noul": 0.7}}}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        seen["body"] = json.loads(req.data.decode())
        return FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    eng = JevK5ServerBackend(base_url="http://k5-test:8090")
    out = eng.systemone(
        "s", [{"name": "ok", "type": "noul", "statement": "Fine?"}],
        images=["https://e.com/a.png"], videos=["https://e.com/v.mp4"])
    assert seen["body"]["images"] == ["https://e.com/a.png"]
    assert seen["body"]["videos"] == ["https://e.com/v.mp4"]
    assert out["ok"]["probability"] == pytest.approx(0.7)


def test_jevk5_preserves_server_legend(monkeypatch):
    from systemone.jevk5_backend import JevK5ServerBackend

    class FakeResp:
        status = 200

        def read(self):
            return json.dumps({"answers": {
                "s": {"type": "score", "level": "hi",
                      "distribution": {"lo": 0.2, "hi": 0.8},
                      "legend": {"lo": "Low.", "hi": "High."}}}}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(
        urllib.request, "urlopen", lambda req, timeout=None: FakeResp())
    eng = JevK5ServerBackend(base_url="http://k5-test:8090")
    out = eng.systemone(
        "s", [{"name": "s", "type": "score", "levels": ["lo", "hi"]}])
    assert out["s"]["legend"] == {"lo": "Low.", "hi": "High."}


def test_rerank_reports_dropped_media_and_legend():
    out = RerankBackend(lambda q, d: 1.0 if d == "hi" else 0.0,
                        model_name="fake").systemone(
        "s", [{"name": "s", "type": "score", "levels": ["lo", "hi"],
               "legend": {"lo": "Low.", "hi": "High."}}],
        images=["a"], videos=["b", "c"])
    assert out["s"]["legend"] == {"lo": "Low.", "hi": "High."}
    assert out["_meta"]["media_dropped"] == {"images": 1, "videos": 2}
