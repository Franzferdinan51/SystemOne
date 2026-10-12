"""systemone-bridge plugin — Hermes <-> SystemOne DecisionLoop.

Hooks:
  ``pre_tool_call``  — confidence-gate high-risk tool calls (deletes,
                       dangerous shell, sensitive-path writes). Raises to
                       block when SystemOne confidence < 0.70. Fail-open.
  ``post_tool_call`` — log every tool call (arg shapes only, never values)
                       to calibration.jsonl for later judge calibration.

Tools (toolset ``systemone``):
  ``systemone_env``    — run one action string in FileSystemEnv / ShellEnv /
                         WebFetchEnv.
  ``systemone_decide`` — answer multiple choice/yes-no questions in one
                         batched call.
  ``systemone_run``    — run a full See → Decide → Act loop for a task.

Command:
  ``/systemone-bridge`` — status | test <action...> | decisions
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from .bridge import CONFIDENCE_FLOOR, ENV_NAMES, SystemOneBridge

logger = logging.getLogger(__name__)

_bridge: Optional[SystemOneBridge] = None


def _bridge_instance() -> SystemOneBridge:
    global _bridge
    if _bridge is None:
        _bridge = SystemOneBridge()
    return _bridge


# ---------------------------------------------------------------------------
# pre_tool_call: risk gating
# ---------------------------------------------------------------------------

_DELETE_TOOLS = {"delete_file", "remove_file", "rm", "delete"}
_SHELL_TOOLS = {"terminal", "exec", "shell", "run_command", "bash"}
_WRITE_TOOLS = {"write_file", "patch", "edit_file"}

_DANGEROUS_CMD = re.compile(
    r"(\brm\b[^;|&]*\s-[a-z]*r|\bsudo\b|\bmkfs\b|\bdd\b[^;|&]*\bof=/dev/"
    r"|\b(shutdown|reboot|halt|poweroff)\b|(curl|wget)[^|;]*\|\s*(ba)?sh"
    r"|chmod\s+(-R\s+)?777|:\(\)\s*\{\s*:\|\:&\s*\}\s*;)",
    re.IGNORECASE,
)
_SENSITIVE_PATH = re.compile(
    r"(\.ssh/|/etc/passwd|/etc/shadow|credentials|\.pem\b|\.key\b|authorized_keys)",
    re.IGNORECASE,
)


def _looks_risky(tool_name: str, args: Dict[str, Any]) -> Optional[str]:
    """Return a risk reason if the call should be gated, else None."""
    name = (tool_name or "").lower()
    if name in _DELETE_TOOLS:
        return f"file deletion via {tool_name}"
    if name in _SHELL_TOOLS:
        cmd = str(args.get("command") or args.get("cmd") or "")
        if _DANGEROUS_CMD.search(cmd):
            return "dangerous shell command"
        return None
    if name in _WRITE_TOOLS:
        path = str(args.get("path") or "")
        if _SENSITIVE_PATH.search(path):
            return "write to sensitive path"
        return None
    return None


def _describe_call(tool_name: str, args: Dict[str, Any]) -> str:
    parts = [f"tool={tool_name}"]
    for key in ("command", "cmd", "path", "file", "target"):
        val = args.get(key)
        if val:
            s = str(val)
            parts.append(f"{key}={s[:300]}")
    return " ".join(parts)


def _on_pre_tool_call(tool_name: str = "", args: Optional[Dict[str, Any]] = None,
                      task_id: str = "", session_id: str = "",
                      tool_call_id: str = "", **_: Any) -> None:
    """Gate high-risk calls through SystemOne. Raises to block; fail-open."""
    try:
        reason = _looks_risky(tool_name, args or {})
        if not reason:
            return
        bridge = _bridge_instance()
        desc = _describe_call(tool_name, args or {})
        approved, confidence, why = bridge.decide_action(
            desc, context=f"risk={reason}")
        if not approved:
            raise RuntimeError(
                f"[systemone-bridge] BLOCKED '{tool_name}' "
                f"(confidence {confidence:.2f} < {CONFIDENCE_FLOOR:.2f}; {reason}). "
                f"{why}. Escalate to System 2 review or run it manually."
            )
        logger.info("systemone-bridge: allowed '%s' (conf %.2f)", tool_name, confidence)
    except RuntimeError:
        raise  # intentional block
    except Exception as exc:  # fail-open: never break Hermes on gate errors
        logger.warning("systemone-bridge: pre_tool_call failed open (%s)", exc)


def _on_post_tool_call(tool_name: str = "", args: Optional[Dict[str, Any]] = None,
                       result: Any = None, task_id: str = "",
                       session_id: str = "", tool_call_id: str = "",
                       **_: Any) -> None:
    """Log tool calls for calibration. Never raises."""
    try:
        _bridge_instance().log_calibration(
            tool_name or "unknown", args or {}, session_id=session_id, ok=True)
    except Exception as exc:
        logger.warning("systemone-bridge: post_tool_call logging failed (%s)", exc)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def _tool_env(args: Optional[Dict[str, Any]] = None, **_: Any) -> str:
    """systemone_env: run one action in a SystemOne Env."""
    a = args or {}
    out = _bridge_instance().run_env_action(
        str(a.get("env", "filesystem")),
        str(a.get("action", "")),
        a.get("start_dir"),
    )
    return out


def _tool_decide(args: Optional[Dict[str, Any]] = None, **_: Any) -> str:
    """systemone_decide: batched choice/yes-no questions, one call."""
    a = args or {}
    questions = a.get("questions") or []
    if not isinstance(questions, list):
        return json.dumps({"error": "questions must be a list"})
    answers = _bridge_instance().batched_decide(
        questions, context=str(a.get("context", "")))
    if not answers:
        return json.dumps({"error": "decision backend unavailable (fail-open: no signal)"})
    return json.dumps(answers, indent=2)


def _tool_run(args: Optional[Dict[str, Any]] = None, **_: Any) -> str:
    """systemone_run: full See → Decide → Act loop for a task."""
    a = args or {}
    task = str(a.get("task", ""))
    if not task:
        return json.dumps({"outcome": "error", "error": "task is required"})
    result = _bridge_instance().run_task(
        task=task,
        env_name=str(a.get("env", "filesystem")),
        candidates=list(a.get("candidates") or []),
        max_ticks=int(a.get("max_ticks", 10) or 10),
        start_dir=a.get("start_dir"),
    )
    return json.dumps(result, indent=2)[:8000]


_TOOLS = {
    "systemone_env": (
        _tool_env,
        "Run one action string in a SystemOne Env (filesystem: ls/cd/read/write/mkdir/rm; "
        "shell: any command with safety blocks; webfetch: fetch <url>). Returns the "
        "observation. Env state persists per (env, start_dir) within the session.",
        {"env": {"type": "string", "enum": list(ENV_NAMES),
                 "description": "Which Env to drive."},
         "action": {"type": "string",
                    "description": "One Env action, e.g. 'ls /tmp', 'read /etc/hosts', 'fetch https://example.com'."},
         "start_dir": {"type": "string",
                       "description": "Working directory for filesystem/shell Envs (default: Hermes cwd)."}},
        ["env", "action"],
    ),
    "systemone_decide": (
        _tool_decide,
        "Answer multiple choice or yes/no questions in ONE batched SystemOne call "
        "(the /v1/decisions pattern). Each question: {name, type: 'choice'|'noul', "
        "options: [...]}. Returns {name: {choice, confidence}}. v1 uses a heuristic "
        "judge; a trained decision model can replace it.",
        {"questions": {"type": "array",
                       "description": "Questions to answer in one batch.",
                       "items": {"type": "object"}},
         "context": {"type": "string",
                     "description": "Shared context/state the questions are about."}},
        ["questions"],
    ),
    "systemone_run": (
        _tool_run,
        "Run a full See → Decide → Act loop for a task using a SystemOne Env. You "
        "propose candidate action strings; the loop picks one per tick with "
        "confidence gating, stall detection and turn memory. Returns the step "
        "transcript and outcome (done/stalled/budget).",
        {"task": {"type": "string", "description": "Task description in plain language."},
         "env": {"type": "string", "enum": list(ENV_NAMES),
                 "description": "Which Env the loop acts in."},
         "candidates": {"type": "array", "items": {"type": "string"},
                        "description": "Candidate action strings to choose from each tick. "
                                       "Include 'done' semantics via the Env's done action."},
         "max_ticks": {"type": "integer",
                       "description": "Max loop ticks (default 10, max 40)."},
         "start_dir": {"type": "string",
                       "description": "Working directory for filesystem/shell Envs."}},
        ["task", "candidates"],
    ),
}


# ---------------------------------------------------------------------------
# Slash command
# ---------------------------------------------------------------------------

_HELP = (
    "systemone-bridge — Hermes <-> SystemOne DecisionLoop\n\n"
    "  /systemone-bridge status     — backend availability\n"
    "  /systemone-bridge test <action...> — run the risk gate on a description\n"
    "  /systemone-bridge decisions  — recent gate decisions\n"
    "  /systemone-bridge decide <json> — batched questions as JSON list\n\n"
    "Tools: systemone_env, systemone_decide, systemone_run (toolset 'systemone')."
)


def _handle_slash(raw_args: str) -> str:
    argv = (raw_args or "").strip().split(None, 1)
    cmd = argv[0] if argv else "status"
    rest = argv[1] if len(argv) > 1 else ""
    bridge = _bridge_instance()
    if cmd == "status":
        return (f"SystemOne available: {bridge.available}\n"
                f"dir: {bridge.systemone_dir}\n"
                f"confidence floor: {CONFIDENCE_FLOOR}\n"
                f"envs: {', '.join(ENV_NAMES)}")
    if cmd == "test":
        if not rest:
            return "Usage: /systemone-bridge test <action description>"
        approved, conf, why = bridge.decide_action(rest)
        return f"approved={approved} confidence={conf:.2f}\n{why}"
    if cmd == "decisions":
        rows = bridge.recent_decisions(10)
        if not rows:
            return "No gate decisions logged yet."
        return "\n".join(
            f"- [{r.get('ts','?')}] approved={r.get('approved')} "
            f"conf={r.get('confidence')} :: {(r.get('action') or '')[:80]}"
            for r in rows)
    if cmd == "decide":
        try:
            questions = json.loads(rest)
        except ValueError as exc:
            return f"Invalid JSON: {exc}"
        answers = bridge.batched_decide(questions)
        return json.dumps(answers, indent=2) if answers else "no signal (backend unavailable)"
    return f"Unknown subcommand: {cmd}\n\n{_HELP}"


def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    for name, (handler, description, properties, required) in _TOOLS.items():
        parameters: Dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            parameters["required"] = required
        ctx.register_tool(
            name=name, toolset="systemone", handler=handler,
            description=description,
            schema={"name": name, "description": description, "parameters": parameters},
        )
    ctx.register_command("systemone-bridge", handler=_handle_slash,
                         description="SystemOne decision bridge: status, gate test, batched decisions.")
