"""JevBench adapter: run SystemOne over JevBench-style typed-decision items.

JevBench (fstandhartinger/jevbench) ranks Jev-class systems on public +
sealed items shaped like::

    {"id": ..., "family": ..., "labels": [...],
     "question": {"type": ..., "instructions": ..., "criteria": {...}},
     "expected": ..., "state": ..., "provenance": {"exclude_reason": ...}}

This module scores such items with any SystemOne engine (duck-typed
``systemone(state, questions)``) and summarizes accuracy. Conventions mirror
JevBench's own adapter base (adapters/base.py): the request never carries
the expected label, probabilities are native distributions (Brier/ECE
eligible — never verbalized), and latency is wall-clock per item.

Two entry points:

- score_item(item, engine): one DecisionResult.
- run_file(path, engine, ...): a whole jsonl split -> summary (+ optional
  predictions file). ``systemone jevbench`` wraps this for the CLI.

Registering upstream (their repo, their harness) would vendor score_item's
mapping into jevbench/adapters/; this module keeps the mapping tested here
so a registration patch is mechanical.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DecisionResult:
    """Per-item outcome, mirroring JevBench adapters/base.py."""

    adapter: str
    ok: bool
    probs: Optional[dict] = None  # native distribution over the answer keys
    probs_source: str = "native"  # "native" | "verbalized" (never logprobs)
    model: str = ""
    error: Optional[str] = None
    latency_s: float = 0.0
    usage: dict = field(default_factory=dict)
    label: Optional[str] = None

    def to_public(self) -> dict:
        """Public-safe view: no raw response text, no request body."""
        return {
            "adapter": self.adapter,
            "ok": self.ok,
            "probs": self.probs,
            "probs_source": self.probs_source,
            "model": self.model,
            "error": self.error,
            "latency_s": self.latency_s,
            "usage": self.usage,
            "label": self.label,
        }


def item_to_body(item: Dict[str, Any]) -> Dict[str, Any]:
    """JevBench item -> TypeSafe-dialect request body (no expected leakage).

    The question key is fixed ('decision'); the expected answer never
    appears in the request. Labels appear only as the answer vocabulary
    the judge must choose among (options/levels), never as the answer.

    Score items whose criteria list holds level *descriptions* get their
    labels as the level names, with descriptions folded into the
    instructions — otherwise bare level names ("0".."3") carry no
    meaning for the judge.
    """
    question = item.get("question") or {}
    qtype = question.get("type")
    instructions = question.get("instructions")
    criteria = question.get("criteria")
    if qtype == "score" and isinstance(criteria, list):
        labels = item.get("labels") or []
        if labels and len(labels) == len(criteria):
            descs = "\n".join(f"{lab}: {c}" for lab, c in zip(labels, criteria))
            level_block = f"Levels:\n{descs}"
            instructions = f"{instructions}\n{level_block}" if instructions else level_block
            criteria = list(labels)
    q: Dict[str, Any] = {"type": qtype, "instructions": instructions}
    if criteria is not None:
        q["criteria"] = criteria
    return {"state": item.get("state"), "questions": {"decision": q}}


def normalize_answer(answer: Dict[str, Any]) -> tuple[Optional[str], Optional[dict]]:
    """Engine answer -> (label, probs), following the djev-adapter mapping.

    - choice: probabilities as-is, label = argmax.
    - noul: {"yes": P(yes), "no": 1-P(yes)}, label = argmax.
    - score: distribution as-is, label = level.
    """
    atype = answer.get("type")
    if atype == "choice":
        probs = dict(answer.get("probabilities") or {})
        label = answer.get("choice") or (max(probs, key=probs.get) if probs else None)
        return label, probs or None
    if atype == "noul":
        try:
            p = float(answer.get("probability", 0.5))
        except (TypeError, ValueError):
            p = 0.5
        p = max(0.0, min(1.0, p))
        probs = {"yes": p, "no": 1.0 - p}
        return ("yes" if p >= 0.5 else "no"), probs
    if atype == "score":
        dist = dict(answer.get("distribution") or {})
        label = answer.get("level") or (max(dist, key=dist.get) if dist else None)
        return label, dist or None
    return None, None


def score_item(
    item: Dict[str, Any],
    engine: Any,
    *,
    adapter: str = "systemone",
    model: str = "",
) -> DecisionResult:
    """Score one JevBench item with a SystemOne engine."""
    if (item.get("provenance") or {}).get("exclude_reason"):
        return DecisionResult(adapter=adapter, ok=False, model=model,
                              error="excluded by provenance")
    from .shim import translate_body

    t0 = time.perf_counter()
    try:
        state_text, questions = translate_body(item_to_body(item))
        answers = engine.systemone(state_text, questions)
        label, probs = normalize_answer(answers.get("decision") or {})
        if label is None:
            raise ValueError("engine returned no decision")
        latency = time.perf_counter() - t0
        return DecisionResult(
            adapter=adapter, ok=True, probs=probs, model=model or getattr(
                engine, "model_name", ""),
            latency_s=latency, label=label,
        )
    except Exception as exc:  # fail-open per item: one bad item never kills a run
        return DecisionResult(
            adapter=adapter, ok=False, model=model,
            error=f"{type(exc).__name__}: {exc}"[:200],
            latency_s=time.perf_counter() - t0,
        )


def run_file(
    path: str,
    engine: Any,
    *,
    out: Optional[str] = None,
    limit: Optional[int] = None,
    adapter: str = "systemone",
    model: str = "",
) -> Dict[str, Any]:
    """Score a JevBench jsonl split; optionally write predictions jsonl.

    Returns a summary: {n, scored, correct, accuracy, mean_latency_ms,
    by_family: {family: {n, correct, accuracy}}, adapter, model}.
    """
    model = model or getattr(engine, "model_name", "")
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if limit is not None:
        rows = rows[: max(0, limit)]

    scored = correct = 0
    lat_total = 0.0
    by_family: Dict[str, Dict[str, Any]] = {}
    predictions: List[Dict[str, Any]] = []
    for item in rows:
        res = score_item(item, engine, adapter=adapter, model=model)
        predictions.append({"id": item.get("id"), **res.to_public()})
        if not res.ok or res.label is None:
            continue
        scored += 1
        lat_total += res.latency_s
        fam = by_family.setdefault(item.get("family", "unknown"),
                                   {"n": 0, "correct": 0})
        fam["n"] += 1
        if res.label == item.get("expected"):
            correct += 1
            fam["correct"] += 1
    for fam in by_family.values():
        fam["accuracy"] = (fam["correct"] / fam["n"]) if fam["n"] else None
    summary = {
        "adapter": adapter,
        "model": model,
        "n": len(rows),
        "scored": scored,
        "correct": correct,
        "accuracy": (correct / scored) if scored else None,
        "mean_latency_ms": round(lat_total / scored * 1000, 1) if scored else None,
        "by_family": by_family,
    }
    if out:
        with open(out, "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(json.dumps(pred) + "\n")
        summary["predictions"] = out
    return summary
