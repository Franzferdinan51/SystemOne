"""systemone-bridge — Hermes <-> SystemOne DecisionLoop bridge.

Bridges Hermes to SystemOne's decision machinery:

* **Gating** (:meth:`SystemOneBridge.decide_action`): confidence-gate risky
  actions before they execute.
* **Batched decisions** (:meth:`SystemOneBridge.batched_decide`): answer
  multiple choice/yes-no questions in one call (the ``/v1/decisions``
  pattern).
* **Env tools** (:meth:`SystemOneBridge.run_env_action`): drive
  ``FileSystemEnv`` / ``ShellEnv`` / ``WebFetchEnv`` as structured tools.
* **Task runner** (:meth:`SystemOneBridge.run_task`): full
  See → Decide → Act loop over an Env from a task description.
* **Calibration logging** (:meth:`SystemOneBridge.log_calibration`): record
  tool calls for later judge calibration.

v1 uses heuristic System 1 judges (risk-pattern and keyword-overlap
scoring, stdlib only) so the plugin works with zero model dependencies.
The loop machinery (StallGuard, confidence gate, turn memory) is real
SystemOne code. A trained decision model can replace the heuristics later
without changing the interface.

Fail-open everywhere: if SystemOne cannot be imported, or anything raises
inside the bridge, actions are allowed through and the reason is logged.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

SYSTEMONE_DIR = os.path.expanduser("~/SystemOne")

#: Below this confidence, the pre_tool_call gate blocks the action.
CONFIDENCE_FLOOR = 0.70

#: Env names exposed as tools.
ENV_NAMES = ("filesystem", "shell", "webfetch")

# ---------------------------------------------------------------------------
# Heuristic System 1 judges (v1, stdlib only)
# ---------------------------------------------------------------------------

#: Risk rules for the approve/block gate: (pattern, weight, label).
_RISK_RULES: List[Tuple[re.Pattern, float, str]] = [
    (re.compile(r"\brm\b[^;|&]*\s-[a-z]*r", re.IGNORECASE), 0.45, "recursive delete"),
    (re.compile(r"\b(sudo|su\b|doas)\b", re.IGNORECASE), 0.30, "privilege escalation"),
    (re.compile(r"\b(mkfs|fdisk|parted)\b", re.IGNORECASE), 0.50, "disk destructive"),
    (re.compile(r"\bdd\b[^;|&]*\bof=/dev/", re.IGNORECASE), 0.50, "raw disk write"),
    (re.compile(r"\b(shutdown|reboot|halt|poweroff)\b", re.IGNORECASE), 0.40, "host power control"),
    (re.compile(r"(curl|wget)[^|;]*\|\s*(ba)?sh", re.IGNORECASE), 0.45, "remote code execution"),
    (re.compile(r"chmod\s+(-R\s+)?777", re.IGNORECASE), 0.35, "world-writable permissions"),
    (re.compile(r"(\.ssh/|/etc/passwd|/etc/shadow|credentials|\.pem\b|\.key\b|authorized_keys)", re.IGNORECASE), 0.25, "sensitive path"),
    (re.compile(r"\b(delete|destroy|drop\s+table|truncate|format)\b", re.IGNORECASE), 0.20, "destructive verb"),
    (re.compile(r"--force|\bforce\b", re.IGNORECASE), 0.15, "force flag"),
    (re.compile(r":\(\)\s*\{\s*:\|\:&\s*\}\s*;", re.IGNORECASE), 0.50, "fork bomb"),
    (re.compile(r">\s*/dev/(sd|hd|nvme)", re.IGNORECASE), 0.50, "device overwrite"),
]


def _gate_judge(state_text: str,
                questions: Sequence[Dict[str, Any]],
                images: Sequence[Any] = (),
                videos: Sequence[Any] = ()) -> Dict[str, Any]:
    """Heuristic approve/block judge: score risk patterns in the state text."""
    text = state_text or ""
    risk = 0.0
    reasons: List[str] = []
    for pattern, weight, label in _RISK_RULES:
        if pattern.search(text):
            risk += weight
            reasons.append(label)
    risk = min(risk, 0.95)
    confidence = round(1.0 - risk, 3)
    choice = "block" if confidence < CONFIDENCE_FLOOR else "approve"
    answers: Dict[str, Any] = {"_meta": {"judge": "heuristic-gate-v1", "reasons": reasons}}
    for q in questions:
        if q.get("name") == "action":
            answers["action"] = {"choice": choice, "confidence": confidence}
    return answers


_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")
_STOPWORDS = frozenset(
    "the and for with from that this what when where which who how are was were "
    "has have had not but all any can will would should could your you its into over "
    "than then them they their there these those".split()
)


def _tokenize(text: str) -> set:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


def _choice_judge(state_text: str,
                  questions: Sequence[Dict[str, Any]],
                  images: Sequence[Any] = (),
                  videos: Sequence[Any] = ()) -> Dict[str, Any]:
    """Heuristic choice/noul judge: keyword overlap between options and state.

    Each option is scored by shared tokens with the state text (which
    carries the task description plus the latest observation). Confidence
    is the winner's share of total score; ties and zero overlap give low
    confidence so the loop's gate can trip.
    """
    state_tokens = _tokenize(state_text or "")
    # Penalize options already tried recently: the task runner appends a
    # "Recently tried" line to the state so the loop explores instead of
    # repeating the same action every tick.
    recent_tokens: set = set()
    m = re.search(r"recently tried.*?:(.*)", state_text or "",
                  re.IGNORECASE | re.DOTALL)
    if m:
        recent_tokens = _tokenize(m.group(1))
    answers: Dict[str, Any] = {"_meta": {"judge": "heuristic-choice-v1"}}
    for q in questions:
        name = q.get("name", "")
        qtype = q.get("type", "choice")
        options = list(q.get("options") or [])
        if qtype == "noul":
            options = ["false", "true"]
        if not options:
            continue
        scored = []
        for opt in options:
            opt_tokens = _tokenize(str(opt))
            overlap = len(state_tokens & opt_tokens)
            repeat = len(opt_tokens & recent_tokens)
            scored.append((overlap - repeat, opt))
        scored.sort(key=lambda p: p[0], reverse=True)
        total = sum(s for s, _ in scored)
        if total <= 0:
            # No signal: pick first option with minimal confidence.
            answers[name] = {"choice": str(scored[0][1]), "confidence": 0.10}
            continue
        top_score, top_opt = scored[0]
        second = scored[1][0] if len(scored) > 1 else 0
        # confidence from winner's share, discounted by closeness of runner-up
        confidence = round((top_score / total) * (top_score / (top_score + second + 1e-9)), 3)
        confidence = max(0.05, min(0.95, confidence))
        answers[name] = {"choice": str(top_opt), "confidence": confidence}
    return answers


# ---------------------------------------------------------------------------
# Env adapters
# ---------------------------------------------------------------------------

class _GateEnv:
    """Dry-run Env for gating: the 'world' is the proposed action description."""

    def __init__(self, description: str, Observation: Any, ActResult: Any) -> None:
        self._description = description
        self._Observation = Observation
        self._ActResult = ActResult

    def observe(self) -> Any:
        return self._Observation(text=self._description, done=False)

    def act(self, action: str) -> Any:
        return self._ActResult(progressed=True, done=True, info={"decision": action})


class _TaskEnv:
    """Wraps a real Env, injecting the task description into each observation."""

    def __init__(self, env: Any, task: str, Observation: Any) -> None:
        self._env = env
        self._task = task
        self._Observation = Observation
        self._recent: List[str] = []

    def observe(self) -> Any:
        obs = self._env.observe()
        text = f"Task: {self._task}\nObservation:\n{obs.text}"
        if self._recent:
            text += ("\nRecently tried (do not repeat unless it progressed): "
                     + ", ".join(self._recent[-3:]))
        return self._Observation(
            text=text,
            images=getattr(obs, "images", ()),
            videos=getattr(obs, "videos", ()),
            done=bool(getattr(obs, "done", False)),
            info=dict(getattr(obs, "info", {}) or {}),
        )

    def act(self, action: str) -> Any:
        self._recent.append(action)
        return self._env.act(action)


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

class SystemOneBridge:
    """Wraps SystemOne's DecisionLoop + Envs for Hermes."""

    def __init__(self,
                 systemone_dir: Optional[str] = None,
                 confidence_floor: float = CONFIDENCE_FLOOR) -> None:
        self.systemone_dir = systemone_dir or SYSTEMONE_DIR
        self.confidence_floor = confidence_floor
        self._available = False
        self._gate_loop: Any = None
        self._task_loop: Any = None
        self._Observation: Any = None
        self._ActResult: Any = None
        self._env_classes: Dict[str, Any] = {}
        self._env_instances: Dict[str, Any] = {}
        self._lock = threading.Lock()
        plugindir = Path(__file__).resolve().parent
        self._decision_log = plugindir / "decisions.log"
        self._calibration_log = plugindir / "calibration.jsonl"
        self._init()

    def _init(self) -> None:
        try:
            if self.systemone_dir not in sys.path:
                sys.path.insert(0, self.systemone_dir)
            from systemone.loop import ActResult, DecisionLoop, Observation
            from systemone import envs as envs_mod
            self._Observation = Observation
            self._ActResult = ActResult
            self._env_classes = {
                "filesystem": envs_mod.FileSystemEnv,
                "shell": envs_mod.ShellEnv,
                "webfetch": envs_mod.WebFetchEnv,
            }
            self._gate_loop = DecisionLoop(
                judge=_gate_judge,
                action_name="action",
                uncertainty_action="block",
                confidence_floor=self.confidence_floor,
                escalate_below=0.0,
                system2=None,
                budget=4,
                system_prompt="You gate risky actions. Approve only safe ones.",
            )
            self._task_loop = DecisionLoop(
                judge=_choice_judge,
                action_name="action",
                uncertainty_action="done",   # low confidence -> stop safely
                confidence_floor=0.35,
                escalate_below=0.0,
                system2=None,
                budget=40,
                system_prompt="You drive a task step by step. Pick the best next action.",
            )
            self._available = True
            logger.info("systemone-bridge: DecisionLoop + Envs ready (dir=%s)", self.systemone_dir)
        except Exception as exc:  # fail-open
            logger.warning("systemone-bridge: SystemOne unavailable (%s); fail-open", exc)
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    # -- gating ----------------------------------------------------------

    def decide_action(self, action_description: str,
                      context: str = "") -> Tuple[bool, float, str]:
        """Gate one proposed action. Never raises (fail-open)."""
        if not self._available:
            return True, 0.0, "fail-open: SystemOne DecisionLoop unavailable"
        try:
            with self._lock:
                state = f"Proposed action: {action_description}"
                if context:
                    state += f"\nContext: {context}"
                env = _GateEnv(state, self._Observation, self._ActResult)
                questions = [{"name": "action", "type": "choice",
                              "options": ["approve", "block"]}]
                result = self._gate_loop.run(env, questions)
            if not result.steps:
                raise RuntimeError("DecisionLoop returned no steps")
            step = result.steps[-1]
            approved = step.action == "approve"
            confidence = float(step.confidence or 0.0)
            reasoning = (
                f"decision={step.action} confidence={confidence:.2f} "
                f"gated={step.gated} ticks={result.n_ticks}"
            )
            self._log_decision(action_description, approved, confidence, reasoning)
            return approved, confidence, reasoning
        except Exception as exc:
            logger.warning("systemone-bridge: gate failed open (%s)", exc)
            return True, 0.0, f"fail-open: gate error ({exc})"

    # -- batched decisions -----------------------------------------------

    def batched_decide(self,
                       questions: Sequence[Dict[str, Any]],
                       context: str = "") -> Dict[str, Dict[str, Any]]:
        """Answer multiple choice/noul questions in one call.

        ``questions``: [{"name":..., "type":"choice"|"noul",
        "options":[...]}, ...]. Returns {name: {"choice", "confidence"}}.
        Never raises: on error returns {} (caller treats as no-signal).
        """
        if not self._available:
            return {}
        try:
            state = context or "Answer the questions."
            answers = _choice_judge(state, list(questions))
            answers.pop("_meta", None)
            return answers
        except Exception as exc:
            logger.warning("systemone-bridge: batched_decide failed (%s)", exc)
            return {}

    # -- env tools --------------------------------------------------------

    def _get_env(self, env_name: str, start_dir: Optional[str] = None) -> Any:
        """Shared Env instance per (env, start_dir); raises if unavailable."""
        if not self._available:
            raise RuntimeError("SystemOne Envs unavailable")
        key = f"{env_name}::{start_dir or ''}"
        with self._lock:
            env = self._env_instances.get(key)
            if env is None:
                cls = self._env_classes[env_name]
                env = cls(start_dir) if env_name in ("filesystem", "shell") else cls()
                self._env_instances[key] = env
            return env

    def run_env_action(self, env_name: str, action: str,
                       start_dir: Optional[str] = None) -> str:
        """Run one action string in a SystemOne Env. Never raises."""
        try:
            if env_name not in ENV_NAMES:
                return f"error: unknown env {env_name!r} (want one of {ENV_NAMES})"
            env = self._get_env(env_name, start_dir)
            with self._lock:
                result = env.act(action)
                obs = env.observe()
            info = getattr(result, "info", {}) or {}
            lines = [
                f"env: {env_name}",
                f"action: {action}",
                f"progressed: {bool(getattr(result, 'progressed', False))}",
                f"done: {bool(getattr(result, 'done', False))}",
            ]
            if info:
                lines.append(f"info: {json.dumps(info)[:1000]}")
            obs_text = getattr(obs, "text", "") or ""
            lines.append(f"observation:\n{obs_text[:4000]}")
            return "\n".join(lines)
        except Exception as exc:
            logger.warning("systemone-bridge: run_env_action failed open (%s)", exc)
            return f"error: env action failed ({exc})"

    # -- task runner ------------------------------------------------------

    def run_task(self, task: str, env_name: str,
                 candidates: Sequence[str],
                 max_ticks: int = 10,
                 start_dir: Optional[str] = None) -> Dict[str, Any]:
        """Run See → Decide → Act loop for a task.

        ``candidates`` are the action strings Hermes proposes; the loop
        picks one per tick via the choice judge, with confidence gating,
        stall detection and turn memory from SystemOne. Never raises:
        failures return {"outcome": "error", ...}.
        """
        try:
            if env_name not in ENV_NAMES:
                return {"outcome": "error",
                        "error": f"unknown env {env_name!r} (want one of {ENV_NAMES})"}
            if not candidates:
                return {"outcome": "error", "error": "no candidate actions provided"}
            env = self._get_env(env_name, start_dir)
            task_env = _TaskEnv(env, task, self._Observation)
            options = list(candidates) + ["done"]
            questions = [{"name": "action", "type": "choice", "options": options}]
            with self._lock:
                self._task_loop.budget = max(1, min(int(max_ticks), 40))
                result = self._task_loop.run(task_env, questions)
            steps = [{
                "tick": s.tick,
                "action": s.action,
                "confidence": round(float(s.confidence or 0.0), 3),
                "progressed": s.progressed,
                "status": s.status,
                "gated": s.gated,
                "observation": (s.observation or "")[:500],
            } for s in result.steps]
            return {
                "outcome": result.outcome,
                "task": task,
                "env": env_name,
                "ticks": result.n_ticks,
                "steps": steps,
                "memory": result.memory[-6:],
            }
        except Exception as exc:
            logger.warning("systemone-bridge: run_task failed (%s)", exc)
            return {"outcome": "error", "error": str(exc)}

    # -- logging ----------------------------------------------------------

    def _log_decision(self, action: str, approved: bool,
                      confidence: float, reasoning: str) -> None:
        try:
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "action": action[:500],
                "approved": approved,
                "confidence": confidence,
                "reasoning": reasoning,
            }
            with open(self._decision_log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except Exception as exc:
            logger.warning("systemone-bridge: could not write decision log (%s)", exc)

    def log_calibration(self, tool_name: str, args: Optional[Dict[str, Any]],
                        session_id: str = "", ok: bool = True,
                        note: str = "") -> None:
        """Record a tool call for later judge calibration.

        Logs arg names and value sizes only — never raw values — so
        secrets in arguments stay out of the calibration set.
        """
        try:
            arg_summary = {}
            for k, v in (args or {}).items():
                s = str(v)
                arg_summary[k] = {"len": len(s), "type": type(v).__name__}
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "tool": tool_name,
                "args": arg_summary,
                "session": session_id,
                "ok": ok,
                "note": note[:200],
            }
            with open(self._calibration_log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except Exception as exc:
            logger.warning("systemone-bridge: could not write calibration log (%s)", exc)

    def recent_decisions(self, n: int = 10) -> List[Dict[str, Any]]:
        """Last n logged gate decisions, newest first."""
        try:
            lines = self._decision_log.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        out = []
        for line in lines[-n:]:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return list(reversed(out))
