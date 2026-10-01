"""Tests for POST /v1/decide + GET /v1/decide/info on the shim.

No model needed — translation is pure and the HTTP tests use stub engines.
"""

import json
import os
import sys
import threading
import urllib.error
import urllib.request

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from systemone.shim import (  # noqa: E402
    Unprocessable,
    decide_option_limit,
    decide_state_parts,
    serve,
    translate_decide_answer,
    translate_decide_body,
)


class StubEngine:
    model_name = "stub"

    def systemone(self, state, questions):
        answers = {}
        for q in questions:
            if q["type"] == "choice":
                opts = q["options"]
                n = len(opts)
                answers[q["name"]] = {
                    "type": "choice", "choice": opts[0],
                    "probabilities": {o: 1.0 / n for o in opts},
                    "confidence": 1.0 / n,
                }
            elif q["type"] == "score":
                lv = q["levels"]
                n = len(lv)
                answers[q["name"]] = {
                    "type": "score", "level": lv[0],
                    "distribution": {x: 1.0 / n for x in lv},
                    "confidence": 1.0 / n,
                }
            else:
                answers[q["name"]] = {
                    "type": "noul", "probability": 0.8, "answer": True,
                    "confidence": 0.8,
                }
        answers["_meta"] = {"model": "stub", "latency_ms": 2.0}
        return answers


def test_state_parts_accept_all_forms():
    text, imgs = decide_state_parts("plain")
    assert (text, imgs) == ("plain", [])
    text, imgs = decide_state_parts(
        ["Photo: ", {"image": "https://e.com/p.jpg"}, " done"])
    assert text == "Photo: \n done"
    assert imgs == ["https://e.com/p.jpg"]
    text, imgs = decide_state_parts([
        {"type": "text", "text": "hi"},
        {"type": "image_url", "image_url": {"url": "data:abc"}},
    ])
    assert (text, imgs) == ("hi", ["data:abc"])
    text, imgs = decide_state_parts({"page": {"url": "https://e.com"}})
    assert text == "URL: https://e.com" and imgs == []


def test_translate_body_defaults():
    kind, text, imgs, q, opts = translate_decide_body({
        "kind": "noul", "state": "s", "question": "Q?"})
    assert (kind, opts) == ("noul", ["false", "true"])
    kind, _, _, _, opts = translate_decide_body({
        "kind": "score", "state": "s", "question": "Q?"})
    assert (kind, opts) == ("score", ["0", "1", "2", "3", "4", "5"])


def test_translate_body_rejects_bad_input():
    with pytest.raises(Unprocessable):
        translate_decide_body({"kind": "bogus", "state": "s", "question": "Q?"})
    with pytest.raises(Unprocessable):
        translate_decide_body({"kind": "choice", "state": "s", "question": "Q?"})
    with pytest.raises(Unprocessable):
        translate_decide_body({"kind": "choice", "state": "s",
                               "question": "Q?", "options": ["one"]})
    with pytest.raises(Unprocessable):
        translate_decide_body({"kind": "choice", "state": "s",
                               "question": "  ", "options": ["a", "b"]})


def test_translate_answer_shapes():
    out = translate_decide_answer("choice", ["a", "b"], {
        "probabilities": {"a": 3.0, "b": 1.0}}, "m", 20.0)
    assert out["options"] == ["a", "b"]
    assert out["probabilities"] == [0.75, 0.25]
    assert out["choice"] == "a" and out["choice_index"] == 0
    assert out["protocol"] == "jev27-bare-v1"
    assert out["elapsed_seconds"] == pytest.approx(0.02)
    out = translate_decide_answer("noul", ["false", "true"],
                                  {"probability": 0.8}, "m", None)
    assert out["probabilities"] == pytest.approx([0.2, 0.8])
    assert out["choice"] == "true"


def _post(url, body):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_serve_decide_choice_roundtrip():
    server = serve(0, engine=StubEngine())
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        status, payload = _post(f"{base}/v1/decide", {
            "kind": "choice", "state": "Customer: charged twice.",
            "question": "Which team?", "options": ["billing", "shipping"]})
        assert status == 200
        assert payload["choice"] == "billing"
        assert payload["probabilities"] == [0.5, 0.5]
        assert payload["model"] == "stub"
        assert "warnings" not in payload
    finally:
        server.shutdown()
        t.join(timeout=5)
        server.server_close()


def test_serve_decide_images_warned_on_text_engine():
    server = serve(0, engine=StubEngine())
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        status, payload = _post(f"{base}/v1/decide", {
            "kind": "noul",
            "state": ["Photo: ", {"image": "https://e.com/p.jpg"}],
            "question": "Is food shown?"})
        assert status == 200
        assert payload["choice"] == "true"
        assert len(payload["warnings"]) == 1
        status, _ = _post(f"{base}/v1/decide", {
            "kind": "noul", "state": [{"image": "https://e.com/p.jpg"}],
            "question": "Is food shown?"})
        assert status == 422
    finally:
        server.shutdown()
        t.join(timeout=5)
        server.server_close()


def test_serve_decide_bad_body_is_422():
    server = serve(0, engine=StubEngine())
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        status, payload = _post(f"{base}/v1/decide", {"kind": "choice"})
        assert status == 422
    finally:
        server.shutdown()
        t.join(timeout=5)
        server.server_close()


def test_serve_decide_info():
    server = serve(0, engine=StubEngine())
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        with urllib.request.urlopen(
                f"{base}/v1/decide/info", timeout=10) as resp:
            payload = json.loads(resp.read())
        assert payload["kinds"] == ["noul", "choice", "score"]
        assert payload["image_support"] is False
        assert payload["backend"] == "custom"
        assert payload["option_limit"] == 255
    finally:
        server.shutdown()
        t.join(timeout=5)
        server.server_close()


def test_option_limits_per_engine():
    from systemone.jev_backend import JevDecideBackend

    jev = JevDecideBackend.__new__(JevDecideBackend)
    assert decide_option_limit(jev, "jev") == 256
    assert decide_option_limit(StubEngine(), "sglang") == 26
    assert decide_option_limit(StubEngine(), "custom") == 255
