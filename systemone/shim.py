"""Local drop-in for TypeSafe's hosted `/v1/systemone` endpoint.

Ryan's `jev-ultrafast` and `mobile-jev` agents POST Jev-shaped bodies to
`https://api.typesafe.ai/v1/systemone` with an API key. This module serves
the same request/response dialect from a local GLiClass engine — no key,
no cloud, no per-call cost.

Run:
    python -m systemone.shim [--port 8765]
    # env: SYSTEMONE_MODEL, SYSTEMONE_DEVICE

Then point the agent at it. In jev-ultrafast's `model.py`, the only change
is the endpoint passed to `post_json`:

    # before
    post_json("https://api.typesafe.ai/v1/systemone", os.environ["TYPESAFE_API_KEY"], body)
    # after
    post_json("http://127.0.0.1:8765/v1/systemone", "local", body)

Endpoints:
    POST /v1/systemone        TypeSafe dialect (see below)
    POST /v1/systemone/route  model router: pick the cheapest sufficient
                              local tier for a task (see below)
    GET  /healthz, /           liveness

Request body for /v1/systemone (TypeSafe dialect):
    {"model": "jev-latest",
     "state": {"page": {"url","title","text"}, "elements": [...],
               "recent_actions": [...]} | "plain text...",
     "questions": {"operation": {"type": "choice",
                                "criteria": {"CLICK": "Click ...", ...},
                                "instructions": {"goal": ..., "rules": ...}},
                   "click_target": {"type": "choice",
                                   "criteria": {"1": {"element": "[1] ...",
                                                      "current_value": ...}, ...},
                                   "instructions": ...}}}

Response:
    {"answers": {"operation": {"type": "choice", "choice": "CLICK",
                              "probabilities": {"CLICK": 0.7, ...},
                              "confidence": 0.7}, ...},
     "model": "<local checkpoint>", "usage": {}}

Request body for /v1/systemone/route:
    {"task": "triage this support ticket for urgency",
     "cost_bias": "economy" | "balanced" | "quality",   # optional, default balanced
     "tiers": ["edge", "base"],                          # optional subset
     "registry": {"edge": {"model_id": ..., "description": ...}, ...}}
        # optional; defaults to the bundled model_registry.json.
        # Tiers with model_id=null are not routable.

Response:
    {"route": {"model_id": "knowledgator/gliclass-edge-v3.0",
               "tier": "edge",
               "rationale": "Task '...' — 'edge' (...) is the cheapest tier
                             rated sufficient under the 'balanced' policy
                             (confidence 0.81).",
               "confidence": 0.81,
               "probabilities": {"edge": 0.81, "base": 0.19},
               "cost_bias": "balanced"},
     "model": "<local checkpoint>", "usage": {},
     "latency_ms": 123.4}

The routing judgment itself is made by the already-loaded local engine
(one batched choice call); the registry is only ever a *catalog* — the
router never loads the routed models. Tiers are capability/cost metadata,
mirroring the Loki autorouter pattern (see examples/demo_autorouter.py).

Every POST decision (both endpoints) is appended as one JSON line to
logs/systemone-shim.log (rotating, 1MB x 4): {"ts", "endpoint",
"latency_ms", "status", "model", ...}. Set SYSTEMONE_LOG_FILE to override
the path (tests use this) or SYSTEMONE_LOG_DISABLE=1 to silence.

Only one local model is ever loaded, same as the rest of the package.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional

from .api import MAX_STATE_CHARS, SystemOne, validate_choice

# -- model router -----------------------------------------------------------

REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "model_registry.json")

COST_BIAS_POLICIES = {
    "economy": "Aggressively prefer the cheapest tier that is still sufficiently capable.",
    "balanced": "Prefer lower cost when capability is sufficient; pay more only for material task needs.",
    "quality": "Prefer capability and reliability, using cost as the tie-breaker among sufficient tiers.",
}


def load_registry(path: str | None = None) -> Dict[str, Dict[str, Any]]:
    """Load the tier registry; returns {tier_name: tier_entry}."""
    with open(path or REGISTRY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    tiers = data.get("tiers", data)
    if not isinstance(tiers, dict) or not tiers:
        raise ValueError("model registry must define a non-empty 'tiers' mapping")
    return tiers


def candidates_from_registry(
    registry: Dict[str, Dict[str, Any]],
    tiers: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """Filter the registry to routable candidates.

    A tier is routable when its entry is a dict with a non-empty string
    `model_id`. Tiers with `model_id: null` (unconfigured, e.g. the 35B
    tier before the user fills in their checkpoint) are skipped.
    Raises ValueError on unknown tier names or zero candidates.
    """
    reg = registry.get("tiers", registry)  # accept both file shape and bare mapping
    if not isinstance(reg, dict):
        raise ValueError("registry must map tier names to tier entries")
    names = list(tiers) if tiers else list(reg.keys())
    unknown = [t for t in names if t not in reg]
    if unknown:
        raise ValueError(f"unknown tier(s): {', '.join(unknown)}")
    candidates: List[Dict[str, str]] = []
    skipped: List[str] = []
    for name in names:
        entry = reg[name]
        if not isinstance(entry, dict):
            raise ValueError(f"registry tier {name!r} must be a mapping")
        model_id = entry.get("model_id")
        if not model_id or not isinstance(model_id, str):
            skipped.append(name)
            continue
        candidates.append(
            {
                "tier": name,
                "model_id": model_id,
                "description": str(entry.get("description", "")),
            }
        )
    if not candidates:
        raise ValueError(
            "no routable tiers: every requested tier has model_id=null "
            "(configure the tier in model_registry.json or pass a registry "
            "with real model ids)"
            + (f"; skipped: {', '.join(skipped)}" if skipped else "")
        )
    return candidates


def build_route_question(
    task: str, candidates: List[Dict[str, str]], cost_bias: str
) -> Dict[str, Any]:
    """A single choice question: which tier should handle this task?"""
    lines = [
        COST_BIAS_POLICIES[cost_bias],
        "",
        "Task:",
        task,
        "",
        "Candidate tiers — choose the cheapest tier sufficient for the task:",
    ]
    for c in candidates:
        lines.append(f"- {c['tier']}: {c['model_id']} — {c['description']}")
    return {
        "name": "route",
        "type": "choice",
        "options": [c["tier"] for c in candidates],
        "prompt": "\n".join(lines),
    }


# ---------------------------------------------------------------------------
# Hybrid routing policy (deterministic signals + GLiClass judge)
# ---------------------------------------------------------------------------
#
# The zero-shot choice head is a weak signal for capability-tier routing: on
# the bundled tiers it returns near-uniform probabilities (~0.44-0.47) no
# matter how the tier descriptions are worded. So the deterministic layer
# below carries the decision, and the classifier acts as a cheap Jev-style
# second opinion that can only *raise* the tier when it is confident
# (>= 0.65). When the two judges confidently disagree, routing confidence
# drops; below 0.60 the router escalates one tier rather than risk
# under-provisioning the task.

_TIER_ORDER = ("economy", "balanced", "heavy")
_ESCALATION_THRESHOLD = 0.60
_CLASSIFIER_RAISE_CONFIDENCE = 0.65
_DISAGREEMENT_PENALTY = 0.12
_LONG_INPUT_CHARS = 2000

# Coarse reasoning-effort hint derived from the routed tier, consumed by
# downstream agent loops (ZCode CLI) to size reasoningLevel/maxOutputTokens.
# economy -> low, balanced -> medium, heavy -> high.
_TIER_EFFORT = {"economy": "low", "balanced": "medium", "heavy": "high"}

# Deterministic keyword labels for the routed task. Deliberately short and
# documented: downstream tool routing matches on these labels, not free text.
# (keyword-group, label) pairs; groups are independent so a task can carry
# several labels.
_TASK_LABEL_KEYWORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("calendar", "meeting", "schedule", "appointment"), "calendar"),
    (("email", "inbox", "gmail"), "email"),
    (("debug", "bug", "stack trace", "race condition", "refactor"), "code"),
    (("summar", "tldr", "recap"), "summarize"),
    (("write", "draft", "compose"), "writing"),
    (("file", "folder", "directory"), "files"),
    (("search", "find", "lookup"), "search"),
    (("image", "photo", "picture", "video"), "media"),
    (("deploy", "server", "docker"), "ops"),
)


def task_labels_for(task: str) -> list:
    """Small deterministic keyword labels for *task* (case-insensitive)."""
    lowered = (task or "").lower()
    return [
        label
        for keywords, label in _TASK_LABEL_KEYWORDS
        if any(keyword in lowered for keyword in keywords)
    ]

# Obvious "this needs the strong model" markers (regexes over lowercased text).
_HEAVY_PATTERNS = [
    r"debug(ging|ger)?s?\b",
    r"deadlock",
    r"race condition",
    r"stack ?trace",
    r"traceback",
    r"segfault",
    r"memory leak",
    r"multithread",
    r"concurren\w*",
    r"distributed",
    r"refactor",
    r"architect(ure)?",
    r"theorem",
    r"\bproof\b",
    r"\bprov(e|ing)\b",
    r"calculus",
    r"\bintegral\b",
    r"differential equation",
    r"linear algebra",
    r"cryptograph",
    r"compiler",
    r"\bkernel\b",
    r"contract",
    r"\blegal\b",
    r"lawsuit",
    r"compliance",
    r"medical",
    r"diagnos(is|ed|ing|tic)\b",
    r"security audit",
    r"vulnerab",
    r"\bexploit\b",
    r"penetration test",
    r"\bai agent\b",
    r"tool[- ]?use\b",
    r"\btool[- ]?call\w*\b",
    r"\bmcp\b",
    r"\bnavigat\w*\b",
    r"\bscrol\w*\b",
    r"\bbrowser (automation|navigat\w*|tool\w*|agent\w*)\b",
    r"system design",
    r"design a (system|distributed)",
    r"roadmap",
    r"performance tun",
]

# Obvious "the tiny model is plenty" markers.
_ECONOMY_PATTERNS = [
    r"summariz",
    r"\bsummary\b",
    r"tl;?dr\b",
    r"one[- ]sentence",
    r"\bextract\b",
    r"classif(y|ication)",
    r"rewrite",
    r"translat",
    r"spell(ing| ?check)?\b",
    r"\bgrammar\b",
    r"proofread",
    r"\bhaiku\b",
    r"\bpoem\b",
    r"capital of",
    r"bullet points",
]

# Ultra-trivial Q&A: bare arithmetic and short factual questions. Single-lookup
# tasks where the cheapest tier is plenty. Kept separate from _ECONOMY_PATTERNS
# because the short-question rule also needs a length + shape check (below).
_TRIVIAL_ARITHMETIC_PATTERNS = [
    r"\d+\s*[+\-*/^]\s*\d+",  # bare arithmetic expression: 2+2, 3 * 4
    r"\bwhat is [\d][\d\s+\-*/().^%]*\??",  # "what is 2+2?"
    r"\bcalculat\w*\b",
    r"\bhow much is\b",
    r"\bhow many\b",
]

# Short factual questions ("What/Who/When/Where/Which ...?") under this length
# are single-lookup tasks -> economy. Heavy patterns still win on conflict
# (substance over form), and the classifier can only raise from here.
_TRIVIAL_QUESTION_MAX_CHARS = 140
_TRIVIAL_QUESTION_STARTERS = ("what", "who", "when", "where", "which")


def analyze_task(task: str) -> Dict[str, Any]:
    """Deterministic complexity analysis: map task text to a suggested tier.

    Returns a dict with the suggested tier name ("economy" | "balanced" |
    "heavy"), a confidence in [0, 1], human-readable reasons, and whether any
    real signal fired (vs. the default middle-tier guess).
    """
    text = (task or "").lower()

    def _hits(patterns: Sequence[str]) -> List[str]:
        found: List[str] = []
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                found.append(match.group(0))
        return found

    heavy_hits = _hits(_HEAVY_PATTERNS)
    economy_hits = _hits(_ECONOMY_PATTERNS)
    if len(task or "") > _LONG_INPUT_CHARS:
        heavy_hits.append(f"long input (>{_LONG_INPUT_CHARS} chars)")

    # Ultra-trivial Q&A shapes: bare arithmetic + short factual questions.
    trivial_hits = _hits(_TRIVIAL_ARITHMETIC_PATTERNS)
    stripped = (task or "").strip()
    words = stripped.split()
    if (
        len(stripped) <= _TRIVIAL_QUESTION_MAX_CHARS
        and stripped.endswith("?")
        and words
        and words[0].lower().rstrip(",") in _TRIVIAL_QUESTION_STARTERS
    ):
        trivial_hits.append(f"short {words[0].lower()}-question")
    if stripped.lower().startswith("define ") and len(stripped) <= _TRIVIAL_QUESTION_MAX_CHARS:
        trivial_hits.append("define-X")
    economy_hits += trivial_hits

    reasons = [f"complexity signal: '{h}'" for h in heavy_hits]
    reasons += [f"trivial-task signal: '{h}'" for h in economy_hits]

    if heavy_hits and economy_hits:
        # Substance wins over form: a legal/medical/technical document that
        # needs summarizing still needs the strong model to get it right.
        reasons.append("conflicting signals; erring toward heavy")
        return {"tier": "heavy", "confidence": 0.55, "reasons": reasons,
                "has_signal": True}
    if len(heavy_hits) >= 2:
        return {"tier": "heavy", "confidence": 0.85, "reasons": reasons,
                "has_signal": True}
    if heavy_hits:
        return {"tier": "heavy", "confidence": 0.62, "reasons": reasons,
                "has_signal": True}
    if economy_hits:
        return {"tier": "economy", "confidence": 0.80, "reasons": reasons,
                "has_signal": True}
    reasons.append("no strong complexity signals; default middle tier")
    return {"tier": "balanced", "confidence": 0.68, "reasons": reasons,
            "has_signal": False}


def parse_route_body(
    body: Dict[str, Any], default_registry: Dict[str, Dict[str, Any]]
) -> tuple[str, str, List[Dict[str, str]]]:
    """Validate a /v1/systemone/route body -> (task, cost_bias, candidates)."""
    task = body.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("request must include a non-empty 'task' string")
    cost_bias = body.get("cost_bias", "balanced")
    if cost_bias not in COST_BIAS_POLICIES:
        raise ValueError(
            f"unknown cost_bias {cost_bias!r}; "
            f"expected one of: {', '.join(COST_BIAS_POLICIES)}"
        )
    registry = body.get("registry") or default_registry
    tiers = body.get("tiers")
    if tiers is not None and (
        not isinstance(tiers, list) or not all(isinstance(t, str) for t in tiers)
    ):
        raise ValueError("'tiers' must be a list of tier-name strings")
    candidates = candidates_from_registry(registry, tiers)
    return task.strip(), cost_bias, candidates


def route_decision(
    engine: Any,
    task: str,
    candidates: List[Dict[str, str]],
    cost_bias: str,
) -> Dict[str, Any]:
    """Pick the cheapest sufficient tier for *task*.

    Hybrid policy:
      1. Deterministic complexity analysis sets the base tier (and a floor
         when real signals fired).
      2. The GLiClass choice head is a second opinion: it may *raise* the
         tier when confident (>= 0.65), never lower it.
      3. cost_bias nudges one tier toward cheap ("economy") or capable
         ("quality"); "economy" never drops below the deterministic floor.
      4. If final confidence is below 0.60, escalate one tier toward
         capability rather than risk under-provisioning.

    Candidate order is capability order (cheapest first); the bundled
    registry lists tiers economy -> balanced -> heavy.

    Returns the {"model_id", "tier", "rationale", "confidence",
    "probabilities", "cost_bias", "deterministic_tier", "signals", "effort",
    "task_labels"} route dict.
    """
    # The registry is a name->entry mapping with no guaranteed key order;
    # sort candidates cheapest-first so the index math below is sound.
    _order = {t: i for i, t in enumerate(_TIER_ORDER)}
    candidates = sorted(candidates, key=lambda c: _order.get(c["tier"], 99))
    tiers = [c["tier"] for c in candidates]
    n = len(tiers)
    if n == 0:
        raise ValueError("no candidate tiers to route over")

    det = analyze_task(task)
    det_rank = _TIER_ORDER.index(det["tier"])
    det_idx = {0: 0, 1: n // 2, 2: n - 1}[det_rank]

    question = build_route_question(task, candidates, cost_bias)
    answers = engine.systemone(task, [question])
    answer = validate_choice(answers["route"], question["options"])
    clf_probs = {tier: float(prob) for tier, prob in answer["probabilities"].items()}
    clf_tier = answer["choice"]
    clf_conf = float(answer["confidence"])
    clf_idx = tiers.index(clf_tier)

    idx = det_idx
    notes = list(det["reasons"])
    if clf_conf >= _CLASSIFIER_RAISE_CONFIDENCE and clf_idx > idx:
        idx = clf_idx
        conf = clf_conf
        notes.append(
            f"classifier raised tier to '{clf_tier}' (confidence {clf_conf:.2f})"
        )
    elif clf_conf >= _CLASSIFIER_RAISE_CONFIDENCE and clf_idx < det_idx:
        conf = max(0.0, det["confidence"] - _DISAGREEMENT_PENALTY)
        notes.append(
            f"classifier disagreed downward ('{clf_tier}', {clf_conf:.2f}); "
            "confidence reduced"
        )
    else:
        conf = det["confidence"] + (0.05 if clf_idx == idx else 0.0)
        conf = min(0.95, conf)
        notes.append(f"classifier chose '{clf_tier}' (confidence {clf_conf:.2f})")

    floor = det_idx if det["has_signal"] else 0
    if cost_bias == "economy":
        new_idx = max(floor, idx - 1)
        if new_idx != idx:
            notes.append(f"'economy' bias shifted tier down to '{tiers[new_idx]}'")
        idx = new_idx
    elif cost_bias == "quality":
        new_idx = min(n - 1, idx + 1)
        if new_idx != idx:
            notes.append(f"'quality' bias shifted tier up to '{tiers[new_idx]}'")
        idx = new_idx

    if conf < _ESCALATION_THRESHOLD and idx < n - 1:
        idx += 1
        conf = _ESCALATION_THRESHOLD
        notes.append("low routing confidence; escalated one tier toward capability")

    # Blended probability distribution for the response: deterministic
    # one-hot (smoothed) carries 0.65, the classifier's head 0.35.
    det_dist = {
        tier: (0.70 if i == det_idx else (0.30 / (n - 1) if n > 1 else 0.0))
        for i, tier in enumerate(tiers)
    }
    blended = {
        tier: 0.65 * det_dist[tier] + 0.35 * clf_probs.get(tier, 0.0)
        for tier in tiers
    }
    total = sum(blended.values()) or 1.0
    blended = {tier: prob / total for tier, prob in blended.items()}

    winner = candidates[idx]
    task_snip = task if len(task) <= 80 else task[:77] + "..."
    rationale = (
        f"Task '{task_snip}' -> '{winner['tier']}' ({winner['model_id']}): "
        + "; ".join(notes)
        + f" (final confidence {conf:.2f})."
    )
    return {
        "model_id": winner["model_id"],
        "tier": winner["tier"],
        "rationale": rationale,
        "confidence": round(conf, 4),
        "probabilities": blended,
        "cost_bias": cost_bias,
        "deterministic_tier": det["tier"],
        "signals": det["reasons"],
        # Effort hint for agent loops: coarse reasoning budget for this task.
        "effort": _TIER_EFFORT.get(winner["tier"], "medium"),
        # Deterministic keyword labels for tool routing (see _TASK_LABEL_KEYWORDS).
        "task_labels": task_labels_for(task),
    }


# -- latency logging --------------------------------------------------------

_log_lock = threading.Lock()
_logger: Optional[logging.Logger] = None


def get_logger() -> Optional[logging.Logger]:
    """Rotating JSON-lines decision log. Env: SYSTEMONE_LOG_FILE override,
    SYSTEMONE_LOG_DISABLE=1 to silence."""
    global _logger
    if os.environ.get("SYSTEMONE_LOG_DISABLE") == "1":
        return None
    with _log_lock:
        if _logger is not None:
            return _logger
        path = os.environ.get("SYSTEMONE_LOG_FILE") or os.path.join(
            os.path.dirname(__file__), "logs", "systemone-shim.log"
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        logger = logging.getLogger("systemone.shim.decisions")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        _logger = logger
        return logger


def log_decision(record: Dict[str, Any]) -> None:
    """Append one JSON decision record; never raises."""
    try:
        logger = get_logger()
        if logger is None:
            return
        rec = {"ts": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        rec.update(record)
        logger.info(json.dumps(rec))
    except Exception:
        pass  # logging must never break serving


# -- TypeSafe dialect translation (unchanged) --------------------------------

def _render_instructions(instructions: Any) -> str:
    """TypeSafe instructions may be a string or a dict (goal/rules)."""
    if instructions is None:
        return ""
    if isinstance(instructions, str):
        return instructions
    if isinstance(instructions, dict):
        parts = []
        if instructions.get("goal"):
            parts.append(f"Goal: {instructions['goal']}")
        rules = instructions.get("rules")
        if rules:
            if isinstance(rules, (list, tuple)):
                parts.append("Rules:\n" + "\n".join(f"- {r}" for r in rules))
            else:
                parts.append(f"Rules: {rules}")
        for k, v in instructions.items():
            if k not in ("goal", "rules"):
                parts.append(f"{k}: {v}")
        return "\n".join(parts)
    return str(instructions)


def _render_criterion(key: str, value: Any) -> str:
    """A criterion may be a plain description or a structured dict."""
    if isinstance(value, str):
        return f"{key}: {value}"
    if isinstance(value, dict):
        bits = [str(value.get("element", key))]
        for field in ("current_value", "value", "role", "checked", "selected", "expanded", "label"):
            if value.get(field) not in (None, ""):
                bits.append(f"{field}={value[field]}")
        return f"{key}: " + " | ".join(bits)
    return f"{key}: {value}"


def translate_question(name: str, q: Dict[str, Any]) -> Dict[str, Any]:
    """TypeSafe question -> systemone question.

    {"type": "choice", "criteria": {opt: desc|{...}}, "instructions": ...}
    becomes {"name", "type": "choice", "options": [...], "prompt": ...}.
    """
    qtype = q.get("type", "choice")
    criteria = q.get("criteria", {}) or {}
    options = list(criteria.keys())
    prompt_bits = [_render_instructions(q.get("instructions"))]
    prompt_bits.append(
        "Options:\n" + "\n".join(_render_criterion(k, v) for k, v in criteria.items())
    )
    prompt = "\n".join(b for b in prompt_bits if b).strip()
    if qtype == "score":
        return {"name": name, "type": "score", "levels": options, "prompt": prompt}
    if qtype == "noul":
        statement = prompt or str(criteria)
        return {"name": name, "type": "noul", "statement": statement}
    return {"name": name, "type": "choice", "options": options, "prompt": prompt}


def state_to_text(state: Any) -> str:
    """TypeSafe state may be a rich dict or plain text; flatten to text."""
    if state is None:
        return ""
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        parts: List[str] = []
        page = state.get("page") or {}
        if isinstance(page, dict):
            if page.get("url"):
                parts.append(f"URL: {page['url']}")
            if page.get("title"):
                parts.append(f"Title: {page['title']}")
            if page.get("text"):
                parts.append(f"Page text: {page['text']}")
        else:
            parts.append(str(page))
        elements = state.get("elements") or []
        if elements:
            lines = []
            for el in elements:
                if isinstance(el, dict):
                    label = el.get("label", el.get("index", "?"))
                    role = el.get("role", "")
                    lines.append(f"[{el.get('index', '?')}] {label} ({role})".strip())
                else:
                    lines.append(str(el))
            parts.append("Elements:\n" + "\n".join(lines))
        actions = state.get("recent_actions") or state.get("history") or []
        if actions:
            lines = []
            for a in actions[-10:]:
                if isinstance(a, dict):
                    lines.append(
                        f"- {a.get('action', a.get('kind', '?'))}: {a.get('text', '')}".strip()
                    )
                else:
                    lines.append(f"- {a}")
            parts.append("Recent actions:\n" + "\n".join(lines))
        return "\n\n".join(parts)
    return str(state)


def translate_body(body: Dict[str, Any]) -> tuple[str, List[Dict[str, Any]]]:
    """Split a TypeSafe request into (state_text, systemone questions)."""
    state_text = state_to_text(body.get("state", ""))
    if len(state_text) > MAX_STATE_CHARS:
        state_text = state_text[:MAX_STATE_CHARS]
    questions = [
        translate_question(name, q)
        for name, q in (body.get("questions") or {}).items()
    ]
    if not questions:
        raise ValueError("request must include at least one question")
    return state_text, questions


def translate_answers(answers: Dict[str, Any]) -> Dict[str, Any]:
    """systemone answers -> TypeSafe {"answers": ...} response body."""
    out: Dict[str, Any] = {}
    for name, ans in answers.items():
        if name == "_meta" or not isinstance(ans, dict):
            continue
        atype = ans.get("type")
        if atype == "choice":
            out[name] = {
                "type": "choice",
                "choice": ans["choice"],
                "probabilities": ans["probabilities"],
                "confidence": ans["confidence"],
            }
        elif atype == "score":
            out[name] = {
                "type": "score",
                "level": ans["level"],
                "distribution": ans["distribution"],
                "confidence": ans["confidence"],
            }
        elif atype == "noul":
            out[name] = {
                "type": "noul",
                "probability": ans["probability"],
                "answer": ans["answer"],
                "confidence": ans["confidence"],
            }
    return out


class ShimHandler(BaseHTTPRequestHandler):
    """HTTP handler; the engine is attached as `server.engine`,
    the tier registry as `server.registry`."""

    server_version = "SystemOneShim/0.2"

    def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    def _handle_systemone(self) -> tuple[int, Dict[str, Any]]:
        """POST /v1/systemone -> (status, payload)."""
        body = self._read_body()
        state_text, questions = translate_body(body)
        answers = self.server.engine.systemone(state_text, questions)
        return 200, {
            "answers": translate_answers(answers),
            "model": self.server.engine.model_name,
            "usage": {},
            "latency_ms": answers.get("_meta", {}).get("latency_ms"),
        }

    def _handle_route(self) -> tuple[int, Dict[str, Any]]:
        """POST /v1/systemone/route -> (status, payload)."""
        body = self._read_body()
        task, cost_bias, candidates = parse_route_body(body, self.server.registry)
        route = route_decision(self.server.engine, task, candidates, cost_bias)
        return 200, {
            "route": route,
            "model": self.server.engine.model_name,
            "usage": {},
        }

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/healthz"):
            self._send_json(200, {"ok": True, "model": self.server.engine.model_name})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        t0 = time.perf_counter()
        status, payload, extra = 500, {"error": "internal"}, {}
        try:
            if self.path == "/v1/systemone":
                status, payload = self._handle_systemone()
                extra = {"n_questions": len(payload.get("answers", {}))}
            elif self.path == "/v1/systemone/route":
                status, payload = self._handle_route()
                route = payload.get("route", {})
                extra = {
                    "route_tier": route.get("tier"),
                    "route_model": route.get("model_id"),
                    "route_confidence": route.get("confidence"),
                }
            else:
                status = 404
                payload = {
                    "error": "not found, POST /v1/systemone or /v1/systemone/route"
                }
        except (ValueError, KeyError) as e:
            status, payload = 400, {"error": f"bad request: {e}"}
        except Exception as e:  # never leak internals beyond the class name
            status, payload = 500, {"error": f"engine failure: {type(e).__name__}"}
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        if status == 200 and "latency_ms" not in payload:
            payload["latency_ms"] = latency_ms
        self._send_json(status, payload)
        log_decision(
            {
                "endpoint": self.path,
                "latency_ms": latency_ms,
                "status": status,
                "model": getattr(self.server.engine, "model_name", "?"),
                **extra,
            }
        )

    def log_message(self, fmt: str, *args: Any) -> None:
        pass  # quiet by default; the decision log records what matters


# -- self-daemonization (Windows sshd job-object escape) --------------------
#
# sshd on Windows runs each session inside a Job Object with
# KILL_ON_JOB_CLOSE. Node's `detached: true` cannot escape that job -- Node
# exposes no way to set process creation flags -- so a shim spawned by the
# ZCode CLI would die when the SSH session closes. The shim instead re-spawns
# *itself* with CREATE_BREAKAWAY_FROM_JOB | DETACHED_PROCESS; the original
# process exits immediately and the detached grandchild (outside the job)
# serves. Non-Windows platforms are unaffected (Node's setsid() already
# detaches there). Every failure path is fail-open: the shim simply keeps
# running in-process.

def _win32_detach(argv: list[str]) -> bool:
    """Re-launch this shim detached on Windows.

    Returns True when the caller must exit immediately (a detached copy was
    started); False when the current process should keep serving -- not on
    Windows, opted out, or the re-spawn failed (fail-open).
    """
    if sys.platform != "win32":
        return False
    try:
        import subprocess

        creationflags = getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0) | getattr(
            subprocess, "DETACHED_PROCESS", 0
        )
        if not creationflags:
            return False
        # --no-daemonize goes last so the grandchild does not respawn again
        # (SYSTEMONE_DAEMONIZE is inherited through the environment).
        cmd = [sys.executable, "-m", "systemone.shim", *argv, "--no-daemonize"]
        subprocess.Popen(
            cmd,
            creationflags=creationflags,
            close_fds=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def serve(
    port: int = 8765,
    engine: SystemOne | None = None,
    registry: Dict[str, Dict[str, Any]] | None = None,
) -> ThreadingHTTPServer:
    """Build (but do not block on) the shim server."""
    engine = engine or SystemOne(model_name=os.environ.get("SYSTEMONE_MODEL"))
    server = ThreadingHTTPServer(("127.0.0.1", port), ShimHandler)
    server.engine = engine  # type: ignore[attr-defined]
    server.registry = registry if registry is not None else load_registry()  # type: ignore[attr-defined]
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Local /v1/systemone shim server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--daemonize",
        action="store_true",
        default=os.environ.get("SYSTEMONE_DAEMONIZE", "").strip() == "1",
        help=(
            "Windows only: re-spawn detached (break away from the sshd job "
            "object) so the shim survives the parent SSH session. Fail-open; "
            "no-op on other platforms."
        ),
    )
    parser.add_argument(
        "--no-daemonize",
        dest="daemonize",
        action="store_false",
        help="Opt out of --daemonize / SYSTEMONE_DAEMONIZE.",
    )
    args = parser.parse_args()
    if args.daemonize and _win32_detach(sys.argv[1:]):
        print("systemone shim detached; parent exiting")
        return
    server = serve(args.port)
    print(
        f"systemone shim on http://127.0.0.1:{args.port}/v1/systemone "
        f"and /v1/systemone/route (model {server.engine.model_name})"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
