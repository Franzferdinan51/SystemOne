"""Tests for systemone.jevbench (JevBench-split scoring adapter).

Stub engine, synthetic items — no weights, no server, no network.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from systemone.jevbench import (  # noqa: E402
    DecisionResult,
    item_to_body,
    normalize_answer,
    run_file,
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
