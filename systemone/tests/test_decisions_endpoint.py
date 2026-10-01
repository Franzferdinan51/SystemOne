"""Tests for POST /v1/decisions on the shim and the SGLang question shape.

No model needed — translation is pure and the HTTP tests use a stub engine.
"""

import json
import os
import sys
import threading
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from systemone.shim import (  # noqa: E402
    decisions_input_to_text,
    serve,
    translate_answers,
    translate_body,
    translate_decisions_answers,
    translate_decisions_body,
    translate_question,
)


class StubEngine:
    model_name = "stub"

    def systemone(self, state, questions, batch_size=32):
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
        answers["_meta"] = {"model": "stub", "latency_ms": 1.0}
        return answers


def _decisions_body():
    return {
        "input": "desktop with 47 icons, mostly screenshots",
        "questions": [
            {"id": "action", "type": "choice",
             "question": "What should the agent do next?",
             "options": [{"name": "up"}, {"name": "down"}, {"name": "wait"}]},
            {"id": "stuck", "type": "yes_no",
             "question": "Is the agent stuck?"},
            {"id": "threat", "type": "score",
             "question": "How messy is the desktop?",
             "levels": [{"name": "0"}, {"name": "1"}, {"name": "2"}]},
        ],
    }


def _post(port, path, body):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, json.loads(resp.read())


def test_translate_decisions_body():
    state, questions, ids = translate_decisions_body(_decisions_body())
    assert state == "desktop with 47 icons, mostly screenshots"
    assert ids == ["action", "stuck", "threat"]
    by_name = {q["name"]: q for q in questions}
    assert by_name["action"]["options"] == ["up", "down", "wait"]
    assert by_name["stuck"]["type"] == "noul"
    assert by_name["threat"]["levels"] == ["0", "1", "2"]


def test_translate_decisions_answers_shape():
    answers = StubEngine().systemone("", [
        {"name": "action", "type": "choice", "options": ["up", "down"]},
        {"name": "stuck", "type": "noul", "statement": "stuck?"},
    ])
    out = translate_decisions_answers(answers, ["action", "stuck"])
    assert out["action"]["type"] == "choice"
    assert out["action"]["choice"] == "up"
    assert out["action"]["label_mass"] is None
    assert out["stuck"]["type"] == "yes_no"
    assert out["stuck"]["answer"] is True


def test_decisions_input_parts():
    parts = [{"type": "text", "text": "hello"},
             {"type": "image_url", "image_url": {"url": "data:..."}}]
    assert decisions_input_to_text(parts) == "hello"
    assert decisions_input_to_text("plain") == "plain"
    assert decisions_input_to_text(None) == ""


def test_http_decisions_roundtrip():
    server = serve(0, engine=StubEngine())
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        status, payload = _post(port, "/v1/decisions", _decisions_body())
        assert status == 200
        assert set(payload["answers"]) == {"action", "stuck", "threat"}
        assert payload["answers"]["action"]["choice"] == "up"
        assert payload["answers"]["stuck"]["type"] == "yes_no"
        assert payload["answers"]["threat"]["type"] == "score"
        assert payload["model"] == "stub"
        assert payload["usage"] == {}
        assert "latency_ms" in payload
    finally:
        server.shutdown()
        t.join(timeout=5)


def test_http_decisions_422_on_too_many_options():
    server = serve(0, engine=StubEngine())
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        body = {"input": "x", "questions": [
            {"id": "q", "type": "choice", "question": "q?",
             "options": [{"name": f"o{i}"} for i in range(27)]}]}
        try:
            _post(port, "/v1/decisions", body)
            raise AssertionError("expected HTTPError 422")
        except urllib.request.HTTPError as e:
            assert e.code == 422
    finally:
        server.shutdown()
        t.join(timeout=5)


def test_http_decisions_422_on_empty_questions():
    server = serve(0, engine=StubEngine())
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        try:
            _post(port, "/v1/decisions", {"input": "x", "questions": []})
            raise AssertionError("expected HTTPError 422")
        except urllib.request.HTTPError as e:
            assert e.code == 422
    finally:
        server.shutdown()
        t.join(timeout=5)


def test_translate_question_accepts_sglang_shape():
    q = translate_question("action", {
        "type": "choice",
        "question": "What next?",
        "options": [{"name": "up"}, "down"],
    })
    assert q["name"] == "action"
    assert q["type"] == "choice"
    assert q["options"] == ["up", "down"]
    # the question text is kept and the options are enumerated for the engine
    assert "What next?" in q["prompt"]
    assert "up" in q["prompt"] and "down" in q["prompt"]
    q2 = translate_question("stuck", {"type": "yes_no", "question": "Stuck?"})
    assert q2 == {"name": "stuck", "type": "noul", "statement": "Stuck?"}


def test_translate_question_keeps_typesafe_shape():
    q = translate_question("operation", {
        "type": "choice",
        "criteria": {"CLICK": "Click it.", "WAIT": "Wait."},
        "instructions": {"goal": "do the thing"},
    })
    assert q["options"] == ["CLICK", "WAIT"]
    assert "do the thing" in q["prompt"]


def test_translate_body_accepts_question_list():
    body = {"state": "s", "questions": [
        {"id": "a", "type": "choice", "question": "q?",
         "options": [{"name": "x"}, {"name": "y"}]},
    ]}
    state, questions = translate_body(body)
    assert state == "s"
    assert questions[0]["options"] == ["x", "y"]


def test_translate_body_dict_shape_still_works():
    body = {"state": "s", "questions": {
        "a": {"type": "choice", "criteria": {"x": "X", "y": "Y"}}}}
    _, questions = translate_body(body)
    assert questions[0]["options"] == ["x", "y"]
    # existing answer translation untouched
    assert translate_answers({}) == {}
