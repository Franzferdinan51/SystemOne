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


class SystemOne:
    """Local System One decision engine.

    Args:
        model_name: HF id of the GLiClass checkpoint. If None, tries
            MODEL_CANDIDATES smallest-first.
        device: "cuda", "cpu", or None (auto).
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
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
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
                answers[q["name"]] = {
                    "type": "choice",
                    "choice": best,
                    "probabilities": prob_map,
                    "confidence": conf,
                }
            elif qtype == "score":
                best = labs[int(probs.argmax())]
                answers[q["name"]] = {
                    "type": "score",
                    "level": best,
                    "distribution": prob_map,
                    "confidence": conf,
                }
            else:  # noul
                p_yes = prob_map["yes"]
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
