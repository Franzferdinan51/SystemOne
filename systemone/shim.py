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

Request body (TypeSafe dialect):
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

Only one local model is ever loaded, same as the rest of the package.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List

from .api import MAX_STATE_CHARS, SystemOne


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
    """HTTP handler; the engine is attached as `server.engine`."""

    server_version = "SystemOneShim/0.1"

    def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/healthz"):
            self._send_json(200, {"ok": True, "model": self.server.engine.model_name})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/systemone":
            self._send_json(404, {"error": "not found, POST /v1/systemone"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            state_text, questions = translate_body(body)
            answers = self.server.engine.systemone(state_text, questions)
            self._send_json(
                200,
                {
                    "answers": translate_answers(answers),
                    "model": self.server.engine.model_name,
                    "usage": {},
                },
            )
        except (ValueError, KeyError) as e:
            self._send_json(400, {"error": f"bad request: {e}"})
        except Exception as e:  # never leak internals; mirror SystemOneError hygiene
            self._send_json(500, {"error": f"engine failure: {type(e).__name__}"})

    def log_message(self, fmt: str, *args: Any) -> None:
        pass  # quiet by default; the engine logs what matters


def serve(port: int = 8765, engine: SystemOne | None = None) -> ThreadingHTTPServer:
    """Build (but do not block on) the shim server."""
    engine = engine or SystemOne(model_name=os.environ.get("SYSTEMONE_MODEL"))
    server = ThreadingHTTPServer(("127.0.0.1", port), ShimHandler)
    server.engine = engine  # type: ignore[attr-defined]
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Local /v1/systemone shim server")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = serve(args.port)
    print(
        f"systemone shim on http://127.0.0.1:{args.port}/v1/systemone "
        f"(model {server.engine.model_name})"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
