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
    """Ask the engine which tier should handle the task.

    Returns the {"model_id", "tier", "rationale", "confidence",
    "probabilities", "cost_bias"} route dict.
    """
    question = build_route_question(task, candidates, cost_bias)
    answers = engine.systemone(task, [question])
    answer = validate_choice(answers["route"], question["options"])
    tier = answer["choice"]
    winner = next(c for c in candidates if c["tier"] == tier)
    conf = float(answer["confidence"])
    task_snip = task if len(task) <= 80 else task[:77] + "..."
    rationale = (
        f"Task '{task_snip}' — '{tier}' ({winner['model_id']}) is the cheapest "
        f"tier rated sufficient under the '{cost_bias}' policy "
        f"(confidence {conf:.2f})."
    )
    return {
        "model_id": winner["model_id"],
        "tier": tier,
        "rationale": rationale,
        "confidence": conf,
        "probabilities": {t: float(p) for t, p in answer["probabilities"].items()},
        "cost_bias": cost_bias,
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
    args = parser.parse_args()
    server = serve(args.port)
    print(
        f"systemone shim on http://127.0.0.1:{args.port}/v1/systemone "
        f"and /v1/systemone/route (model {server.engine.model_name})"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
