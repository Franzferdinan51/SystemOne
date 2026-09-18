"""Tests for the /v1/systemone shim. No model needed — translation is pure,
and the HTTP test uses a stub engine."""

import json
import os
import sys
import threading
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from systemone.shim import (
    serve,
    state_to_text,
    translate_answers,
    translate_body,
    translate_question,
)


def _typesafe_body():
    return {
        "model": "jev-latest",
        "state": {
            "page": {
                "url": "https://example.com/checkout",
                "title": "Checkout",
                "text": "Your cart is empty. Payment method declined.",
            },
            "elements": [
                {"index": "1", "label": "Retry payment", "role": "button"},
                {"index": "2", "label": "Card number", "role": "textbox"},
            ],
            "recent_actions": [{"action": "Opened checkout", "kind": "nav", "text": ""}],
        },
        "questions": {
            "operation": {
                "type": "choice",
                "criteria": {
                    "CLICK": "Click an element.",
                    "TYPE_TEXT": "Enter text in a field.",
                },
                "instructions": {"goal": "Complete checkout", "rules": ["Don't repeat steps."]},
            },
            "click_target": {
                "type": "choice",
                "criteria": {
                    "1": {"element": "[1] Retry payment", "current_value": "", "role": "button"},
                    "2": {"element": "[2] Card number", "current_value": "", "role": "textbox"},
                },
                "instructions": "Choose the best target for CLICK.",
            },
        },
    }


def test_translate_question_choice():
    q = translate_question(
        "operation",
        {"type": "choice",
         "criteria": {"CLICK": "Click it.", "WAIT": "Wait."},
         "instructions": "Pick one."},
    )
    assert q["name"] == "operation"
    assert q["type"] == "choice"
    assert q["options"] == ["CLICK", "WAIT"]
    assert "Click it." in q["prompt"] and "Pick one." in q["prompt"]


def test_translate_question_structured_criteria():
    q = translate_question(
        "click_target",
        {"type": "choice",
         "criteria": {"1": {"element": "[1] Retry payment", "role": "button"}},
         "instructions": None},
    )
    assert "[1] Retry payment" in q["prompt"]
    assert "role=button" in q["prompt"]


def test_translate_question_string_instructions():
    q = translate_question("op", {"criteria": {"A": "do a"}, "instructions": "Just pick."})
    assert q["type"] == "choice"  # default type
    assert "Just pick." in q["prompt"]


def test_state_to_text_dict():
    text = state_to_text(_typesafe_body()["state"])
    assert "https://example.com/checkout" in text
    assert "Retry payment" in text
    assert "Opened checkout" in text


def test_state_to_text_plain_string():
    assert state_to_text("hello") == "hello"
    assert state_to_text(None) == ""


def test_translate_body_splits():
    state_text, questions = translate_body(_typesafe_body())
    assert len(state_text) > 0
    assert [q["name"] for q in questions] == ["operation", "click_target"]
    assert all(q["type"] == "choice" for q in questions)


def test_translate_answers_choice_shape():
    out = translate_answers(
        {"op": {"type": "choice", "choice": "CLICK",
                "probabilities": {"CLICK": 0.7, "WAIT": 0.3}, "confidence": 0.7},
         "_meta": {"latency_ms": 5}}
    )
    assert out == {"op": {"type": "choice", "choice": "CLICK",
                          "probabilities": {"CLICK": 0.7, "WAIT": 0.3},
                          "confidence": 0.7}}


class StubEngine:
    model_name = "stub"

    def systemone(self, state, questions, batch_size=32):
        answers = {}
        for q in questions:
            opts = q["options"]
            n = len(opts)
            probs = {o: 1.0 / n for o in opts}
            answers[q["name"]] = {
                "type": "choice", "choice": opts[0],
                "probabilities": probs, "confidence": 1.0 / n,
            }
        answers["_meta"] = {"model": "stub"}
        return answers


def test_http_roundtrip_with_stub_engine():
    server = serve(0, engine=StubEngine())
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/systemone",
            data=json.dumps(_typesafe_body()).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read())
        assert resp.status == 200
        assert set(payload["answers"]) == {"operation", "click_target"}
        op = payload["answers"]["operation"]
        assert op["choice"] == "CLICK"
        assert abs(sum(op["probabilities"].values()) - 1.0) < 1e-9
        assert payload["model"] == "stub"
        assert payload["usage"] == {}

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=10) as resp:
            assert json.loads(resp.read())["ok"] is True
    finally:
        server.shutdown()
        t.join(timeout=5)


def test_http_bad_path_404():
    server = serve(0, engine=StubEngine())
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/nope",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=10)
            raise AssertionError("expected HTTPError")
        except urllib.request.HTTPError as e:
            assert e.code == 404
    finally:
        server.shutdown()
        t.join(timeout=5)
