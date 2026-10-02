"""Tests for RotationAveraged (test-time option-order averaging).

Stub engines only — no weights, no server.
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from systemone.rotation import RotationAveraged  # noqa: E402


class PositionBiasedEngine:
    """A judge that always favors the FIRST option (position bias)."""

    model_name = "biased"

    def systemone(self, state, questions):
        answers = {}
        for q in questions:
            if q["type"] == "choice":
                opts = q["options"]
                n = len(opts)
                answers[q["name"]] = {
                    "type": "choice", "choice": opts[0],
                    "probabilities": {o: (0.7 if i == 0 else 0.3 / (n - 1))
                                      for i, o in enumerate(opts)},
                    "confidence": 0.5,
                }
            else:
                answers[q["name"]] = {"type": "noul", "probability": 0.9,
                                      "answer": True, "confidence": 0.9}
        answers["_meta"] = {"model": "biased", "latency_ms": 1.0}
        return answers


def test_rotation_averaging_kills_position_bias():
    eng = RotationAveraged(PositionBiasedEngine(), rotations=2)
    out = eng.systemone("s", [{"name": "c", "type": "choice",
                               "options": ["a", "b"]}])
    # biased engine alone would say a=0.7; averaged over both orders: 50/50
    assert out["c"]["probabilities"]["a"] == pytest.approx(0.5)
    assert out["c"]["probabilities"]["b"] == pytest.approx(0.5)
    assert out["_meta"]["rotations"] == 2


def test_non_choice_questions_pass_through():
    eng = RotationAveraged(PositionBiasedEngine(), rotations=3)
    out = eng.systemone("s", [{"name": "n", "type": "noul",
                               "statement": "fine?"}])
    assert out["n"]["probability"] == pytest.approx(0.9)
    assert out["_meta"]["rotations"] == 1  # no choice questions: 1 pass


def test_rotations_capped_by_widest_question():
    eng = RotationAveraged(PositionBiasedEngine(), rotations=9)
    out = eng.systemone("s", [{"name": "c", "type": "choice",
                               "options": ["a", "b", "c"]}])
    assert out["_meta"]["rotations"] == 3


def test_rotated_orders_are_cyclic():
    qs = [{"name": "c", "type": "choice", "options": ["a", "b", "c"]},
          {"name": "n", "type": "noul", "statement": "x"}]
    assert RotationAveraged.rotated(qs, 1)[0]["options"] == ["b", "c", "a"]
    assert RotationAveraged.rotated(qs, 2)[0]["options"] == ["c", "a", "b"]
    assert RotationAveraged.rotated(qs, 1)[1]["statement"] == "x"


def test_needs_two_rotations():
    with pytest.raises(ValueError):
        RotationAveraged(PositionBiasedEngine(), rotations=1)


def test_geometric_mean_math():
    eng = RotationAveraged(PositionBiasedEngine(), rotations=2)
    out = eng.systemone("s", [{"name": "c", "type": "choice",
                               "options": ["a", "b", "c"]}])
    # rotation 0: a=.7,b=.15,c=.15; rotation 1 (b,c,a): b=.7,c=.15,a=.15
    # geometric mean per key, renormalized
    g = {k: math.exp(sum(math.log(v) for v in vs) / 2)
         for k, vs in {"a": [0.7, 0.15], "b": [0.15, 0.7],
                       "c": [0.15, 0.15]}.items()}
    total = sum(g.values())
    for k, v in g.items():
        assert out["c"]["probabilities"][k] == pytest.approx(v / total)
