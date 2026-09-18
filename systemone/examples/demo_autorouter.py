"""Loki Autorouter pattern, local edition.

Loki's "Loki Autorouter (powered by Jev)" classifies the FIRST task of a new
session and routes it to the cheapest sufficiently-capable model in the
currently-selected gateway. It is:
  - session sticky (route once, keep it for the session),
  - fail-open (any failure -> keep the current model),
  - gated by a confidence threshold (default 0.55),
  - guided by a cost_bias policy (economy | balanced | quality),
  - gateway-bounded (never crosses provider/credential boundaries),
  - state-limited (only the first task + model metadata are judged).

This demo mirrors that exact shape with zero API keys: gliclass-edge plays
Jev's role (the tiny, fast decision model), and the "gateway catalog" is the
local GLiClass checkpoint list. Only the first task and model metadata go
into the routing state — no conversation history. One model is loaded at a
time: the router runs on edge, is released, then the routed model is loaded
for the real work.

Session rules (mirroring Loki's `_is_new_root_session`):
  - routes at most once per process;
  - child sessions (SYSTEMONE_PARENT_SESSION set) never re-route;
  - an explicit --force-model always wins over routing and cache;
  - the sticky cache is bounded (200 entries, oldest evicted).

Run:
    python examples/demo_autorouter.py "summarize this 20-page contract"
    python examples/demo_autorouter.py --cost-bias quality "triage this ticket"
    python examples/demo_autorouter.py --cost-bias economy "is this spam?"
    python examples/demo_autorouter.py --force-model knowledgator/gliclass-base-v1.0 "task"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from systemone import SystemOne

# -- the local "gateway catalog": model -> structured capability string -----
# Mirrors Loki's JevAutoCandidate.description(): capability tokens, context,
# and cost — crisp machine-readable facts instead of prose, so the routing
# judgment reads the same information Jev does. Local "cost" is latency and
# memory tier (no per-token pricing on your own hardware).
CANDIDATE_DESCRIPTIONS = {
    "knowledgator/gliclass-edge-v3.0": (
        "text-classification; no-tools; context=512 tokens; "
        "latency=tier-1 (fastest, ~100ms CPU for 3 questions); memory=~0.1GB; "
        "cost=cheapest"
    ),
    "knowledgator/gliclass-small-v1.0": (
        "text-classification; no-tools; context=512 tokens; "
        "latency=tier-2; memory=~0.4GB; cost=moderate"
    ),
    "knowledgator/gliclass-base-v1.0": (
        "text-classification; no-tools; context=512 tokens; "
        "latency=tier-3 (slowest); memory=~0.8GB; cost=highest; "
        "best accuracy on nuanced or ambiguous judgments"
    ),
}

# -- routing policy text, one per cost_bias (same three Loki offers) ---------
POLICIES = {
    "economy": (
        "Aggressively prefer the cheapest model that is still sufficiently capable."
    ),
    "balanced": (
        "Prefer lower cost when capability is sufficient; "
        "pay more only for material task needs."
    ),
    "quality": (
        "Prefer capability and reliability, using cost as the tie-breaker "
        "among sufficient models."
    ),
}

CONFIDENCE_THRESHOLD = 0.55
MAX_TASK_CHARS = 6000
ROUTE_CACHE = os.path.join(os.path.dirname(__file__), "autorouter_route_cache.json")
# Bound the sticky-route cache so it cannot grow without limit (Loki bounds
# its candidate catalog the same way: 12 default, 24 max).
ROUTE_CACHE_MAX_ENTRIES = 200

# In-process guard: never route twice in one process, mirroring Loki's
# `_jev_auto_route_attempted` flag.
_ROUTE_ATTEMPTED = False

# Child sessions never re-route: if SYSTEMONE_PARENT_SESSION is set, this
# process is a sub-session and keeps the parent's route. Mirrors Loki's
# `_is_new_root_session` check on `_parent_session_id`.
_PARENT_SESSION = os.environ.get("SYSTEMONE_PARENT_SESSION", "").strip()


def _clamp_threshold(value: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return CONFIDENCE_THRESHOLD


def _fingerprint(task: str) -> str:
    return hashlib.sha256(task.encode("utf-8", errors="replace")).hexdigest()[:16]


def _load_sticky(fp: str) -> dict | None:
    if not os.path.exists(ROUTE_CACHE):
        return None
    try:
        with open(ROUTE_CACHE, "r", encoding="utf-8") as f:
            return json.load(f).get(fp)
    except Exception:
        return None


def _save_sticky(fp: str, record: dict) -> None:
    cache: dict = {}
    if os.path.exists(ROUTE_CACHE):
        try:
            with open(ROUTE_CACHE, "r", encoding="utf-8") as f:
                cache = json.load(f) or {}
        except Exception:
            cache = {}
    cache[fp] = record
    # Evict oldest entries first (insertion order) to bound the file.
    while len(cache) > ROUTE_CACHE_MAX_ENTRIES:
        cache.pop(next(iter(cache)))
    with open(ROUTE_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def route_first_task(
    task: str,
    current_model: str,
    router: SystemOne,
    cost_bias: str = "balanced",
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> tuple[str, float, str]:
    """Route the first task of a session, Loki-Autorouter style.

    Returns (chosen_model, confidence, reason). Fail-open: on any error the
    current model is kept.
    """
    task = task.strip()[:MAX_TASK_CHARS]
    if not task:
        return current_model, 0.0, "empty task"

    state = {
        "task": task,
        "current_model": current_model,
        "routing_goal": (
            "Minimize local inference cost and latency without selecting "
            "a model too weak for the task."
        ),
    }
    capability_lines = "\n".join(
        f"- {mid}: {desc}" for mid, desc in CANDIDATE_DESCRIPTIONS.items()
    )
    questions = [
        {
            "name": "route_model",
            "type": "choice",
            "options": list(CANDIDATE_DESCRIPTIONS.keys()),
            "prompt": (
                "Choose exactly one model from the options for this agent task. "
                "The model must be sufficiently capable for the task, including "
                "accuracy, context, or latency needs when those are relevant. "
                + POLICIES[cost_bias]
                + "\nCandidate capabilities:\n"
                + capability_lines
            ),
        }
    ]

    try:
        # The router only ever sees the first task + model metadata,
        # exactly like Loki's Jev call (state is small and bounded).
        out = router.systemone(json.dumps(state), questions)
        answer = out["route_model"]
    except Exception as exc:  # fail-open: keep the current model
        print(f"[autorouter] routing failed open ({exc}); keeping {current_model}")
        return current_model, 0.0, "routing error (fail-open)"

    choice = answer["choice"]
    confidence = float(answer["confidence"])
    if choice not in CANDIDATE_DESCRIPTIONS:
        return current_model, confidence, "unknown model returned (fail-open)"
    if confidence < confidence_threshold:
        return current_model, confidence, (
            f"confidence {confidence:.2f} < threshold {confidence_threshold} (fail-open)"
        )
    return choice, confidence, "routed"


def main() -> None:
    global _ROUTE_ATTEMPTED
    parser = argparse.ArgumentParser(description="Local Loki-Autorouter-style model routing")
    parser.add_argument("task", nargs="?", default="triage this support ticket: my invoice is wrong",
                        help="the session's first task to route")
    parser.add_argument("--cost-bias", choices=["economy", "balanced", "quality"],
                        default="balanced")
    parser.add_argument("--current", default="knowledgator/gliclass-small-v1.0",
                        help="the currently selected model")
    parser.add_argument("--no-sticky", action="store_true",
                        help="ignore the session-sticky route cache")
    parser.add_argument("--force-model", default=None,
                        help="explicit model choice; takes precedence over routing "
                             "and cache (explicit user choices always win)")
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD,
                        help="confidence threshold below which routing fails open "
                             f"(default {CONFIDENCE_THRESHOLD})")
    args = parser.parse_args()

    threshold = _clamp_threshold(args.threshold)
    task = args.task
    fp = _fingerprint(task)

    # Explicit user choice always takes precedence over any route.
    if args.force_model:
        chosen, confidence, reason = args.force_model, 1.0, "explicit user choice"
        print(f"[autorouter] explicit model: {chosen}")
    elif _PARENT_SESSION:
        # Child session: never re-route, keep the parent's model.
        chosen, confidence, reason = args.current, 0.0, (
            f"child session (parent={_PARENT_SESSION}); keeping current model"
        )
        print(f"[autorouter] {reason}")
    elif _ROUTE_ATTEMPTED:
        chosen, confidence, reason = args.current, 0.0, (
            "already routed once in this process; keeping current model"
        )
        print(f"[autorouter] {reason}")
    else:
        sticky = None if args.no_sticky else _load_sticky(fp)
        if sticky:
            chosen, confidence = sticky["chosen_model"], sticky["confidence"]
            reason = "session-sticky route reused"
            print(f"[autorouter] sticky route for task {fp}: {chosen} "
                  f"(confidence {confidence:.2f})")
        else:
            # The router model is the smallest/cheapest — it plays Jev's role.
            print("[autorouter] loading router model (gliclass-edge)...")
            router = SystemOne(model_name="knowledgator/gliclass-edge-v3.0")
            chosen, confidence, reason = route_first_task(
                task, args.current, router,
                cost_bias=args.cost_bias, confidence_threshold=threshold,
            )
            # One model at a time: release the router before loading the worker.
            del router
            record = {
                "mode": "local_auto",
                "task_fingerprint": fp,
                "task_preview": task[:120],
                "original_model": args.current,
                "chosen_model": chosen,
                "confidence": round(confidence, 4),
                "confidence_threshold": threshold,
                "cost_bias": args.cost_bias,
                "candidate_count": len(CANDIDATE_DESCRIPTIONS),
                "reason": reason,
            }
            _save_sticky(fp, record)
            print(f"[autorouter] {reason}: {args.current} -> {chosen} "
                  f"(confidence {confidence:.2f})")
        _ROUTE_ATTEMPTED = True

    # Now do real work on the routed model — sticky for the session.
    print(f"[session] loading {chosen} for the session...")
    worker = SystemOne(model_name=chosen)
    sample = [
        {"name": "is_urgent", "type": "noul",
         "statement": "Does this task require urgent handling?"},
        {"name": "complexity", "type": "score",
         "levels": ["trivial", "moderate", "complex", "expert"],
         "prompt": "How complex is this task to resolve?"},
    ]
    out = worker.systemone(task, sample)
    print(f"  urgent P(yes) = {out['is_urgent']['probability']:.3f}")
    print(f"  complexity    = {out['complexity']['level']} "
          f"(confidence {out['complexity']['confidence']:.3f})")
    print(f"  meta: {out['_meta']}")


if __name__ == "__main__":
    main()
