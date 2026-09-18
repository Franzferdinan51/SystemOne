"""Jev-style System One API over local GLiClass models.

One call takes a `state` (text) plus multiple typed questions and answers
them all in a single batched forward pass — adding questions barely changes
latency, mirroring Jev's parallel evaluation.

Question types (mirroring Jev's three primitives):
- choice: pick from a label list -> label + per-option probabilities + confidence
- score:  rate against ordered levels  -> level + distribution + confidence
- noul:   yes/no question              -> probability the statement is true

Only ONE local model is ever loaded per SystemOne instance.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from transformers import AutoTokenizer

from gliclass import GLiClassModel
from gliclass.pipeline import ZeroShotClassificationPipeline

from .calibration import TemperatureCalibrator, softmax

# Smallest-first candidates; the first that loads wins.
MODEL_CANDIDATES = [
    "knowledgator/gliclass-edge-v3.0",
    "knowledgator/gliclass-small-v1.0",
    "knowledgator/gliclass-base-v1.0",
]

# States larger than this are capped before inference (same bound Loki uses
# for the routing task). The encoder truncates to 512 tokens anyway, so the
# cap only bounds memory/log noise — it does not change judgments.
MAX_STATE_CHARS = 6000


def default_device() -> str:
    """Best torch device for this machine: CUDA > Apple MPS > CPU.

    Stock pip torch wheels are CUDA-enabled on Linux/Windows and
    MPS-capable on macOS, so Apple Silicon gets GPU acceleration with no
    extra installs. Never raises: if the MPS backend is absent (older
    torch), it is simply skipped.
    """
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class SystemOneError(RuntimeError):
    """Sanitized engine failure.

    Mirrors Loki's TypeSafeRequestError philosophy: the message is safe to
    surface to callers and logs. It never echoes environment-provided
    secrets, absolute paths, or transport internals — only what went wrong
    and what to do about it.
    """

    def __init__(self, message: str, *, hint: str = "") -> None:
        self.hint = hint
        super().__init__(f"{message} {hint}".strip() if hint else message)


def _sanitize_detail(err: Exception) -> str:
    """One-line, path/secret-free summary of an unexpected exception."""
    text = f"{type(err).__name__}: {err}".splitlines()[0]
    # strip anything that looks like a filesystem path or URL with creds
    return text[:300]


# ---------------------------------------------------------------------------
# Decision patterns ported from Ryan's jev-ultrafast / mobile-jev agent repos.
# Both are TypeSafe-hosted agent apps; what transfers is not their transport
# but their battle-tested decision-engineering discipline.
# ---------------------------------------------------------------------------

import math


def validate_distribution(
    prob_map: Dict[str, float], ids: Sequence[str], best: str, tol: float = 0.02
) -> Dict[str, float]:
    """Response-contract check on a probability distribution.

    Port of jev-ultrafast's ``validate_choice`` (model.py): asserts the best
    label is one of the ids, the probability keys exactly match the ids, every
    value is finite in [0, 1], the values sum to ~1, and the best label actually
    holds the max probability. Raises SystemOneError on any violation —
    catches degenerate or malformed model outputs before they become decisions.
    """
    try:
        numbers = list(prob_map.values())
        valid = (
            best in ids
            and set(prob_map) == set(ids)
            and all(
                isinstance(n, (int, float)) and math.isfinite(n) and 0 <= n <= 1
                for n in numbers
            )
            and abs(sum(numbers) - 1.0) < tol
            and prob_map[best] >= max(numbers) - 1e-6
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise SystemOneError(
            "model returned an invalid decision distribution",
            hint="keys must match the question options, values must be "
            "finite probabilities summing to ~1, and the chosen label must "
            "hold the max probability",
        )
    return prob_map


def validate_choice(answer: Dict[str, Any], ids: Sequence[str], tol: float = 0.02) -> Dict[str, Any]:
    """Validate a Jev-shaped choice answer: {choice, probabilities, confidence}.

    Thin wrapper over validate_distribution matching jev-ultrafast's shape.
    """
    try:
        probs = answer["probabilities"]
        choice = answer["choice"]
    except (KeyError, TypeError):
        raise SystemOneError("model returned a malformed choice answer") from None
    validate_distribution(probs, ids, choice, tol=tol)
    return answer


ABSTAIN_LABEL = "none"


def with_abstain(options: Sequence[str], label: str = ABSTAIN_LABEL) -> List[str]:
    """Append an explicit abstain option to a choice question.

    Port of mobile-jev's NONE pattern (policy.mjs): never force the model to
    pick when nothing fits — "If the desired value is missing, select NONE."
    """
    opts = list(options)
    if label not in opts:
        opts.append(label)
    return opts


class StallGuard:
    """Fail-fast loop/stall detector for decision-driven agents.

    Port of jev-ultrafast's executor discipline (agent.py): consecutive
    no-progress observations trip a stall instead of letting an agent spin
    forever. Call observe() after each executed decision.
    """

    def __init__(self, max_stalls: int = 3) -> None:
        self.max_stalls = max_stalls
        self.stalls = 0

    def observe(self, progressed: bool) -> str:
        """Record whether the last decision made progress.

        Returns "ok", or "stalled" once max_stalls consecutive no-progress
        observations accumulate.
        """
        self.stalls = 0 if progressed else self.stalls + 1
        return "stalled" if self.stalls >= self.max_stalls else "ok"

    def reset(self) -> None:
        self.stalls = 0


class LatencyStats:
    """Tiny p50/p95 latency aggregator for bench and calibration runs.

    Port of mobile-jev's metrics.mjs stats(): SystemOne already reports
    per-call latency_ms in _meta; this aggregates them across runs.
    """

    def __init__(self) -> None:
        self.values: List[float] = []

    def add(self, ms: float) -> None:
        if isinstance(ms, (int, float)) and math.isfinite(ms):
            self.values.append(float(ms))

    def summary(self) -> Dict[str, Any]:
        vals = sorted(self.values)
        n = len(vals)
        if not n:
            return {"count": 0, "median_ms": None, "p95_ms": None, "mean_ms": None}
        mid = n // 2
        median = vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2
        return {
            "count": n,
            "median_ms": round(median, 1),
            "p95_ms": round(vals[math.ceil(n * 0.95) - 1], 1),
            "mean_ms": round(sum(vals) / n, 1),
        }


class SystemOne:
    """Local System One decision engine.

    Args:
        model_name: HF id of the GLiClass checkpoint. If None, tries
            MODEL_CANDIDATES smallest-first.
        device: "cuda", "mps", "cpu", or None / "auto" (auto-detect:
            CUDA if available, else Apple MPS, else CPU).
        temperature: softmax temperature for output probabilities (1.0 = raw).
        calibrator: optional fitted TemperatureCalibrator; overrides temperature.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        temperature: float = 1.0,
        calibrator: TemperatureCalibrator | None = None,
    ) -> None:
        if device is None or (
            isinstance(device, str) and device.strip().lower() == "auto"
        ):
            # "auto" (the SYSTEMONE_DEVICE default) resolves here, so every
            # entry point — shim, CLI, MCP server — gets CUDA > MPS > CPU.
            device = default_device()
        self.device = device

        candidates = [model_name] if model_name else MODEL_CANDIDATES
        last_err: Exception | None = None
        for cand in candidates:
            try:
                self.model = GLiClassModel.from_pretrained(cand)
                self.tokenizer = AutoTokenizer.from_pretrained(cand)
                self.model_name = cand
                last_err = None
                break
            except Exception as e:  # try next candidate
                last_err = e
        if last_err is not None:
            raise SystemOneError(
                "could not load any GLiClass model",
                hint=f"last error: {_sanitize_detail(last_err)}; "
                "check network access to huggingface.co or set a cached model via SYSTEMONE_MODEL",
            )

        self.pipeline = ZeroShotClassificationPipeline(
            self.model, self.tokenizer, device=device
        )
        self.temperature = temperature
        self.calibrator = calibrator

    # -- calibration ----------------------------------------------------
    def set_calibrator(self, calibrator: TemperatureCalibrator) -> None:
        """Attach a fitted TemperatureCalibrator (overrides temperature)."""
        self.calibrator = calibrator

    def _probs(self, scores: np.ndarray) -> np.ndarray:
        if self.calibrator is not None and getattr(self.calibrator, "fitted_", False):
            return np.asarray(self.calibrator.predict_proba([scores])[0])
        T = self.temperature if self.temperature > 0 else 1.0
        return softmax(np.asarray(scores, dtype=np.float64) / T)

    # -- single batched call --------------------------------------------
    def raw_scores(
        self,
        texts: List[str],
        label_lists: List[List[str]],
        prompts: List[str | None] | None = None,
        batch_size: int = 32,
    ) -> List[Dict[str, float]]:
        """One pipeline call; returns per-text {label: raw_score} dicts."""
        results = self.pipeline(
            texts,
            label_lists,
            threshold=0.0,
            batch_size=batch_size,
            classification_type="single_label",
            prompt=prompts,
        )
        out: List[Dict[str, float]] = []
        for res, labs in zip(results, label_lists):
            # res: list of {"label":..., "score":...}; be defensive about
            # threshold filtering by defaulting missing labels to 0.0
            got = {r["label"]: float(r["score"]) for r in res} if res else {}
            out.append({lab: got.get(lab, 0.0) for lab in labs})
        return out

    def systemone(
        self,
        state: str,
        questions: Sequence[Dict[str, Any]],
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """Answer multiple typed questions about `state` in one batched pass.

        Each question: {"name": str, "type": "choice"|"score"|"noul", ...}
          choice: {"options": [str, ...], "prompt": optional str}
          score:  {"levels": [str, ...],  "prompt": optional str}  (ordered)
          noul:   {"statement": str}  (yes/no question about the state)

        Returns {name: answer_dict, ..., "_meta": {...}}.
        """
        questions = list(questions)
        if not questions:
            raise SystemOneError("questions must be non-empty")

        state_capped = False
        if len(state) > MAX_STATE_CHARS:
            state = state[:MAX_STATE_CHARS]
            state_capped = True

        label_lists: List[List[str]] = []
        prompts: List[str | None] = []
        for q in questions:
            qtype = q["type"]
            if qtype == "choice":
                label_lists.append(list(q["options"]))
                prompts.append(q.get("prompt"))
            elif qtype == "score":
                label_lists.append(list(q["levels"]))
                prompts.append(q.get("prompt"))
            elif qtype == "noul":
                label_lists.append(["yes", "no"])
                prompts.append(q.get("statement") or q.get("prompt"))
            else:
                raise SystemOneError(
                    f"unknown question type: {qtype!r}",
                    hint="expected one of: choice, score, noul",
                )

        t0 = time.perf_counter()
        score_dicts = self.raw_scores(
            [state] * len(questions), label_lists, prompts=prompts,
            batch_size=batch_size,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        answers: Dict[str, Any] = {}
        for q, labs, sdict in zip(questions, label_lists, score_dicts):
            scores = np.array([sdict[lab] for lab in labs], dtype=np.float64)
            probs = self._probs(scores)
            prob_map = {lab: float(p) for lab, p in zip(labs, probs)}
            conf = float(probs.max())
            qtype = q["type"]
            if qtype == "choice":
                best = labs[int(probs.argmax())]
                validate_distribution(prob_map, labs, best)
                answers[q["name"]] = {
                    "type": "choice",
                    "choice": best,
                    "probabilities": prob_map,
                    "confidence": conf,
                }
            elif qtype == "score":
                best = labs[int(probs.argmax())]
                validate_distribution(prob_map, labs, best)
                answers[q["name"]] = {
                    "type": "score",
                    "level": best,
                    "distribution": prob_map,
                    "confidence": conf,
                }
            else:  # noul
                p_yes = prob_map["yes"]
                best = "yes" if p_yes >= 0.5 else "no"
                validate_distribution(prob_map, ["yes", "no"], best)
                answers[q["name"]] = {
                    "type": "noul",
                    "probability": p_yes,
                    "answer": bool(p_yes >= 0.5),
                    "confidence": float(max(p_yes, 1.0 - p_yes)),
                }

        answers["_meta"] = {
            "model": self.model_name,
            "device": self.device,
            "n_questions": len(questions),
            "latency_ms": round(latency_ms, 1),
            "state_chars": len(state),
            "state_capped": state_capped,
        }
        return answers

    def speculative_decide(
        self,
        state: str,
        operation: Dict[str, Any],
        targets: Dict[str, Dict[str, Any]],
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """Decide an operation AND its argument in a single batched pass.

        Speculative multi-head pattern ported from jev-ultrafast (model.py)
        and mobile-jev (policy.mjs): one request carries the operation choice
        plus one target choice-head per operation. Only the target head
        selected by the chosen operation is validated and used — *unused
        target heads cannot cause an action*.

        Args:
            state: the state text to decide about.
            operation: a choice question, e.g.
                {"name": "op", "type": "choice",
                 "options": ["click", "fill", "wait"]}
            targets: {operation_value: choice question} for operations that
                take a target, e.g. {"click": {"name": "click_target",
                "type": "choice", "options": ["btn-1", "btn-2"]}}.
                Operations without an entry need no target.

        Returns {"operation", "target" (or None), "confidence",
                 "target_confidence" (or None), "probabilities",
                 "operation_probabilities", "_meta"}.
        """
        op_name = operation.get("name", "operation")
        op_options = list(operation["options"])
        questions: List[Dict[str, Any]] = [operation]
        for op in op_options:
            if op in targets:
                questions.append(targets[op])

        answers = self.systemone(state, questions, batch_size=batch_size)

        op_answer = validate_choice(answers[op_name], op_options)
        op_choice = op_answer["choice"]

        target = None
        target_conf: float | None = None
        target_probs: Dict[str, float] = {}
        if op_choice in targets:
            tq = targets[op_choice]
            t_name = tq.get("name", f"{op_choice}_target")
            t_options = list(tq["options"])
            t_answer = validate_choice(answers[t_name], t_options)
            target = t_answer["choice"]
            target_conf = t_answer["confidence"]
            target_probs = t_answer["probabilities"]

        return {
            "operation": op_choice,
            "target": target,
            "confidence": op_answer["confidence"],
            "target_confidence": target_conf,
            "probabilities": target_probs,
            "operation_probabilities": op_answer["probabilities"],
            "_meta": answers["_meta"],
        }


def make_questions(
    choices: Dict[str, List[str]] | None = None,
    scores: Dict[str, List[str]] | None = None,
    nouls: Dict[str, str] | None = None,
) -> List[Dict[str, Any]]:
    """Convenience builder for question lists."""
    qs: List[Dict[str, Any]] = []
    for name, options in (choices or {}).items():
        qs.append({"name": name, "type": "choice", "options": options})
    for name, levels in (scores or {}).items():
        qs.append({"name": name, "type": "score", "levels": levels})
    for name, statement in (nouls or {}).items():
        qs.append({"name": name, "type": "noul", "statement": statement})
    return qs
