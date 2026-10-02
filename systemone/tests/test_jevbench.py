"""Tests for systemone.jevbench (JevBench-split scoring adapter).

Stub engine, synthetic items — no weights, no server, no network.
"""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from systemone.jevbench import (  # noqa: E402
    DecisionResult,
    RemoteEndpoint,
    item_to_body,
    normalize_answer,
    run_file,
    run_remote,
    score_item,
)


class StubEngine:
    model_name = "stub"

    def __init__(self, first=True):
        self.first = first  # pick first option/level, else last

    def systemone(self, state, questions, batch_size=32):
        answers = {}
        for q in questions:
            if q["type"] == "choice":
                opts = q["options"]
                pick = opts[0] if self.first else opts[-1]
                n = len(opts)
                answers[q["name"]] = {
                    "type": "choice", "choice": pick,
                    "probabilities": {o: 1.0 / n for o in opts},
                    "confidence": 0.5,
                }
            elif q["type"] == "score":
                lv = q["levels"]
                pick = lv[0] if self.first else lv[-1]
                n = len(lv)
                answers[q["name"]] = {
                    "type": "score", "level": pick,
                    "distribution": {x: 1.0 / n for x in lv},
                    "confidence": 0.5,
                }
            else:
                p = 0.9 if self.first else 0.1
                answers[q["name"]] = {
                    "type": "noul", "probability": p, "answer": p >= 0.5,
                    "confidence": 0.5,
                }
        answers["_meta"] = {"model": "stub", "latency_ms": 1.0}
        return answers


def _item(qtype="noul", expected="yes", labels=("no", "yes"), family="policy",
          excluded=False):
    return {
        "id": f"t-{qtype}", "family": family, "labels": list(labels),
        "question": {
            "type": qtype,
            "instructions": "answer the question",
            "criteria": {"a": "A", "b": "B"} if qtype != "noul" else {
                "true": "T", "false": "F"},
        },
        "expected": expected, "state": "some state",
        "provenance": {"exclude_reason": "dup" if excluded else None},
    }


def test_item_to_body_never_leaks_expected():
    body = item_to_body(_item())
    assert set(body["questions"]) == {"decision"}
    assert body["questions"]["decision"]["type"] == "noul"
    assert "expected" not in json.dumps(body)
    assert "labels" not in json.dumps(body)
    assert body["questions"]["decision"].get("criteria")


def test_score_item_noul_correct_and_wrong():
    assert score_item(_item(expected="yes"), StubEngine(first=True)).label == "yes"
    res = score_item(_item(expected="yes"), StubEngine(first=False))
    assert res.label == "no" and res.ok is True
    assert res.probs == {"yes": 0.1, "no": 0.9}
    assert res.probs_source == "native"


def test_item_to_body_score_description_lists():
    item = {
        "id": "s", "family": "f", "labels": ["0", "1"],
        "question": {"type": "score", "instructions": "Count.",
                     "criteria": ["none", "some"]},
        "expected": "1", "state": "s",
        "provenance": {},
    }
    body = item_to_body(item)
    q = body["questions"]["decision"]
    assert q["criteria"] == ["0", "1"]  # labels become the levels
    assert "0: none" in q["instructions"] and "Count." in q["instructions"]
    assert "expected" not in json.dumps(body)


def test_score_item_choice_and_score():
    choice = _item(qtype="choice", expected="a", labels=("a", "b"))
    res = score_item(choice, StubEngine(first=True))
    assert (res.label, res.ok) == ("a", True)
    assert set(res.probs) == {"a", "b"}
    score = _item(qtype="score", expected="b", labels=("a", "b"))
    res = score_item(score, StubEngine(first=False))
    assert (res.label, res.ok) == ("b", True)


def test_score_item_excluded_and_engine_failure():
    res = score_item(_item(excluded=True), StubEngine())
    assert res.ok is False and res.error == "excluded by provenance"

    class Boom:
        model_name = "boom"

        def systemone(self, *a, **k):
            raise RuntimeError("kaput")

    res = score_item(_item(), Boom())
    assert res.ok is False and "kaput" in (res.error or "")


def test_normalize_answer_unknown_type():
    assert normalize_answer({"type": "mystery"}) == (None, None)


def test_run_file_summary_and_predictions(tmp_path):
    items = tmp_path / "items.jsonl"
    items.write_text("\n".join(json.dumps(it) for it in [
        _item(expected="yes", family="policy"),
        _item(expected="no", family="policy"),
        _item(expected="yes", family="safety", excluded=True),
    ]))
    out = tmp_path / "pred.jsonl"
    summary = run_file(str(items), StubEngine(first=True), out=str(out))
    assert summary["n"] == 3
    assert summary["scored"] == 2 and summary["correct"] == 1
    assert summary["accuracy"] == 0.5
    assert summary["by_family"]["policy"] == {
        "n": 2, "correct": 1, "accuracy": 0.5}
    assert "safety" not in summary["by_family"]  # excluded item unscored
    assert summary["predictions"] == str(out)
    preds = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(preds) == 3 and preds[0]["label"] == "yes"


def test_decision_result_public_view():
    res = DecisionResult(adapter="systemone", ok=True, label="yes")
    assert res.to_public() == {
        "adapter": "systemone", "ok": True, "probs": None,
        "probs_source": "native", "model": "", "error": None,
        "latency_s": 0.0, "usage": {}, "label": "yes",
    }


def test_normalize_answer_remote_shapes():
    # Kev/SGLang serve noul as bare {"noul"} and score as index-keyed
    # probabilities + a legend; both normalize onto local shapes.
    label, probs = normalize_answer({"type": "noul", "noul": 0.9})
    assert label == "yes"
    assert probs["yes"] == pytest.approx(0.9)
    assert probs["no"] == pytest.approx(0.1)
    label, probs = normalize_answer({
        "type": "score", "score": 0.7,
        "probabilities": {"0": 0.3, "1": 0.7},
        "legend": {"0": "low", "1": "high"}})
    assert label == "high"
    assert probs == {"low": 0.3, "high": 0.7}


class _StubHandler(BaseHTTPRequestHandler):
    mode = "ok"  # ok | flaky-then-ok | refuse | forbidden
    calls = 0

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).calls += 1
        mode = type(self).mode
        if mode == "forbidden":
            return self._send(401, {"detail": "nope"})
        if mode == "refuse":
            return self._send(422, {"detail": "too long"})
        if mode == "flaky-then-ok" and type(self).calls == 1:
            return self._send(500, {"detail": "boom"})
        q = body["questions"]["decision"]
        if q["type"] == "choice":
            opts = list(q["criteria"])
            ans = {"type": "choice", "choice": opts[0],
                   "probabilities": {o: 1.0 / len(opts) for o in opts}}
        elif q["type"] == "score":
            ans = {"type": "score", "score": 0.0,
                   "probabilities": {"0": 1.0}, "legend": {"0": "low"}}
        else:
            ans = {"type": "noul", "noul": 0.9}
        return self._send(200, {"model": "stub-remote",
                                "answers": {"decision": ans}})

    def log_message(self, *a):
        pass


def _serve_stub(mode="ok"):
    _StubHandler.mode = mode
    _StubHandler.calls = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_run_remote_against_stub_server(tmp_path):
    items = tmp_path / "items.jsonl"
    items.write_text("\n".join(json.dumps(it) for it in [
        _item(expected="yes"), _item(expected="no"),
    ]))
    server, thread = _serve_stub("ok")
    try:
        port = server.server_address[1]
        summary = run_remote(str(items), f"http://127.0.0.1:{port}")
    finally:
        server.shutdown()
        thread.join(timeout=5)
    assert summary["scored"] == 2 and summary["correct"] == 1
    assert summary["served_model"] == "stub-remote"


def test_run_remote_concurrency_preserves_order(tmp_path):
    items = tmp_path / "items.jsonl"
    items.write_text("\n".join(
        json.dumps(_item(expected="yes")) for _ in range(6)))
    server, thread = _serve_stub("ok")
    try:
        port = server.server_address[1]
        out = tmp_path / "pred.jsonl"
        summary = run_remote(str(items), f"http://127.0.0.1:{port}",
                             out=str(out), concurrency=4)
    finally:
        server.shutdown()
        thread.join(timeout=5)
    assert summary["scored"] == 6
    preds = [json.loads(line) for line in out.read_text().splitlines()]
    assert [p["label"] for p in preds] == ["yes"] * 6


def test_remote_retries_500_then_scores(tmp_path):
    items = tmp_path / "items.jsonl"
    items.write_text(json.dumps(_item(expected="yes")))
    server, thread = _serve_stub("flaky-then-ok")
    try:
        port = server.server_address[1]
        summary = run_remote(str(items), f"http://127.0.0.1:{port}")
    finally:
        server.shutdown()
        thread.join(timeout=5)
    assert summary["scored"] == 1 and _StubHandler.calls == 2


def test_remote_422_is_per_item_error(tmp_path):
    items = tmp_path / "items.jsonl"
    items.write_text(json.dumps(_item(expected="yes")))
    server, thread = _serve_stub("refuse")
    try:
        port = server.server_address[1]
        summary = run_remote(str(items), f"http://127.0.0.1:{port}")
    finally:
        server.shutdown()
        thread.join(timeout=5)
    assert summary["scored"] == 0
    assert _StubHandler.calls == 1  # refused, not retried


def test_remote_401_fails_fast(tmp_path):
    import pytest

    items = tmp_path / "items.jsonl"
    items.write_text("\n".join(
        json.dumps(_item(expected="yes")) for _ in range(3)))
    server, thread = _serve_stub("forbidden")
    try:
        port = server.server_address[1]
        with pytest.raises(RuntimeError, match="HTTP 401"):
            run_remote(str(items), f"http://127.0.0.1:{port}")
    finally:
        server.shutdown()
        thread.join(timeout=5)
    assert _StubHandler.calls == 1  # stopped at once, not per item


def test_remote_endpoint_rejects_non_http():
    import pytest

    with pytest.raises(ValueError):
        RemoteEndpoint("file:///etc/passwd")
