# systemone — local, open System One decisions

A Jev-style typed-decision layer over **local** GLiClass checkpoints. TypeSafe's
Jev is a closed API that returns typed decisions with calibrated probabilities
(`choice`, `score`, `noul`). This package rebuilds that shape on your own
hardware: tiny open models (Apache-2.0, 32M–439M params), one batched call,
honest probabilities.

## Map to Jev's primitives

| Jev primitive | systemone | Returns |
|---|---|---|
| Choice | `{"type": "choice", "options": [...]}` | label + per-option probabilities + confidence |
| Score | `{"type": "score", "levels": [...]}` | level + distribution + confidence |
| Noul | `{"type": "noul", "statement": "..."}` | P(statement true) + yes/no answer |

Jev trains calibration in with RLCD. We do the practical equivalent:
**post-hoc calibration** (temperature / Platt / isotonic) fit on your own
labeled data — see `calibration.py`.

## Quickstart

```python
from systemone import SystemOne, make_questions

eng = SystemOne()  # loads gliclass-edge, ONE model at a time

out = eng.systemone("Payment service 500 errors for 12 minutes, no ack from on-call", [
    {"name": "team", "type": "choice",
     "options": ["backend", "frontend", "devops", "support"]},
    {"name": "impact", "type": "score",
     "levels": ["low", "medium", "high", "critical"]},
    {"name": "page_cto", "type": "noul",
     "statement": "Should the CTO be woken up?"},
])

out["team"]    # {"choice": "backend", "probabilities": {...}, "confidence": 0.91}
out["impact"]  # {"level": "critical", "distribution": {...}, "confidence": 0.83}
out["page_cto"]# {"probability": 0.72, "answer": True, "confidence": 0.72}
```

All questions in one call are evaluated in **one batched forward pass** —
adding questions barely changes latency (see `examples/demo_systemone.py`).

### With calibration

```python
from systemone.calibration import TemperatureCalibrator, CalibrationExample

# fit on a few dozen labeled examples of YOUR task
cal = TemperatureCalibrator().fit(scores, labels)
eng.set_calibrator(cal)   # systemone() now returns calibrated probabilities
```

## Pieces

- **`api.py`** — `SystemOne`: loads one GLiClass checkpoint (edge → small → base,
  smallest-first), `systemone(state, questions)` batched inference.
  `SystemOneError` is the single sanitized exception type (mirrors Loki's
  `TypeSafeRequestError`: safe to surface in logs and tool output, never
  echoes env secrets or paths). States over `MAX_STATE_CHARS` (6000, same
  bound Loki uses) are capped with `state_capped: true` in `_meta`.
- **`cli.py`** — local equivalent of Loki's `/jev status`: `python -m
  systemone.cli status` (config health check, `--load` to verify a real
  model load + latency probe) and `python -m systemone.cli ask --state ...
  --questions q.json` (one-shot Jev-style judgments, same question shape
  as the MCP tool).
- **`calibration.py`** — `TemperatureCalibrator`, `PlattCalibrator`,
  `IsotonicCalibrator`, `CalibratedScorer`, `expected_calibration_error()`.
- **`mcp_server.py`** — MCP tools over stdio for agent stacks:
  `typesafe_ask` (Jev-compatible `state` + `questions` interface with
  `{"id", "type", "instructions", "criteria"}` questions and
  `{"answers": {id: ...}}` responses — no API key needed),
  plus domain tools `verify_claims`, `screen_content`, `rank_candidates`.
  Run: `python -m systemone.mcp_server` (env: `SYSTEMONE_MODEL`, `SYSTEMONE_DEVICE`,
  optional `SYSTEMONE_CALIBRATOR`). Only one model is loaded at a time; a per-call
  `model` on `typesafe_ask` swaps the loaded checkpoint.
- **`distill.py`** — label with a teacher (`SyntheticTeacher`, `HFTeacher`,
  `LMStudioTeacher` — one local model at a time), write training JSON,
  fine-tune the edge student via the repo's `train.py`.
- **`tune.py`** — `make_training_json()` + `tune()` wrappers around `train.py`
  for domain fine-tuning on your own decision data.

## Examples

- `examples/demo_systemone.py` — all three primitives + batching timings
- `examples/demo_calibration.py` — ECE before/after on a hand-labeled set
- `examples/demo_autorouter.py` — Loki-Autorouter-style session-sticky model
  routing: gliclass-edge (as the tiny decision model) routes the session's
  first task to the cheapest sufficiently-capable local checkpoint
  (`--cost-bias economy|balanced|quality`, confidence-gated, fail-open)
- `examples/demo_distill.py` — distill a content-safety classifier into gliclass-edge
- `examples/sample_decision_data.json` — routing + guardrail records in tune() format

## Design notes

- **Jev-compatible `typesafe_ask`**: mirrors the interface Loki exposes for
  TypeSafe Jev (`state`, `questions[{id, type, instructions, criteria}]`,
  optional `model` → `{"answers": {id: {choice|score|noul, probabilities,
  confidence}}}`), but runs entirely on the local engine — no
  `TYPESAFE_API_KEY`, no network. Choice criteria maps option names to
  descriptions; score criteria is an ordered level list (the returned `score`
  is the probability-weighted fractional position, matching Jev's semantics);
  noul criteria optionally defines `true`/`false` descriptions. Batch
  independent questions over the same state in one call.
- **Autorouter pattern**: `demo_autorouter.py` mirrors Loki's Jev Auto router —
  `state = {task, current_model, routing_goal}`, one `choice` question over a
  bounded candidate catalog with structured capability/cost descriptions
  (Loki's `capabilities; context=N; cost=$x/M` format, localized to
  latency/memory tiers), one of three cost-bias policies, a confidence
  threshold (0.55, fail-open below it, clamped to [0,1]), and a session-sticky
  route cache keyed by task fingerprint (bounded at 200 entries, oldest
  evicted). Session rules mirror Loki's `_is_new_root_session`: routes at
  most once per process, child sessions (`SYSTEMONE_PARENT_SESSION`) never
  re-route, and an explicit `--force-model` always wins. The catalog is fixed
  to local checkpoints, so routing never crosses a network/credential boundary.

## Designing good questions

Distilled from TypeSafe's docs and Loki's `typesafe-ai` skill — these apply
to `systemone()`, the MCP tools, and the CLI equally:

- **Atomic questions, composed in code.** Each question should ask one
  specific, well-scoped thing — the kind of judgment a knowledgeable person
  could make in a few seconds given the right context. If your question
  needs extended reasoning or weighs several independent factors, decompose
  it: ask each factor as its own question, then combine the results with
  logic in *your* code. When priorities shift, change a coefficient in code
  rather than rewriting a prompt.
- **Keep exact rules, calculations, permissions, and final actions in
  ordinary code.** The judge returns probabilities; code decides what they
  mean. Never ask the judge to fetch secrets or external resources — give it
  only the state required for the judgment.
- **Batch independent questions over the same state.** Every question is
  evaluated in parallel and in isolation; adding questions barely changes
  latency and does not create context-rot.
- **Give each question enough relevant state** — but bound it (6000 chars).
  A judge starved of context returns flat, low-confidence distributions.
- **`noul` 0.5 means "unsure", not "medium intensity."** It is P(yes), 0–1.
  Do not reinterpret it as a strength dial.
- **Don't discard probabilities when a top answer exists.** Preserve the
  full distribution and confidence; let application policy — not the
  argmax — decide thresholds, escalation, retries, or human review.

## Confidence-gated behavior

TypeSafe's confidence model, adapted for local use:

- **Confidence is the shape of the distribution, collapsed to 0–1.**
  Concentrated on one outcome = confident; spread out = uncertain. (Noul's
  confidence is just max(P(yes), P(no)).)
- **Low confidence is diagnostic.** On a choice it usually means none of the
  options is a clear winner; on a score it means the levels are ambiguous,
  multi-dimensional, or the state doesn't contain enough to go on. Treat
  "I don't know" as a useful signal, not a failure.
- **Three paths:** high confidence → act automatically; medium → proceed
  with caution (confirm, flag for review, gather more information); low →
  do not act (route to a human, clarify, or fall back). Where you draw the
  boundaries depends on the stakes.
- **Thresholds scale with risk — there is no single number.** A destructive
  operation is gated higher than a read-only one. 0.5 is the floor that
  catches genuine uncertainty; your code encodes the risk tolerance above it.

## Fail-open & trust boundaries

Principles ported from Loki's Jev integration:

- **The decider never crosses a trust boundary.** Loki's router may only
  choose within the already-selected gateway — it can never move a session
  to different credentials. Our equivalent: the decision layer can never
  grant capabilities, only select among pre-approved ones. Routing and
  decision tools are separate capabilities; enabling one never implies the
  other.
- **Fail-open at every stage, with logging.** Any routing/judgment failure
  keeps the current model or the safe default and proceeds — never hard-fails
  the session. Thresholds are clamped to [0,1]; unknown model names fall
  back to current.
- **Explicit user choices take precedence** over routed or cached ones.
- **Bound everything:** state (6000 chars), candidate catalog (12 default,
  24 max), sticky cache (200 entries). Unbounded inputs are how quiet
  degradation starts.

## Tests

```bash
pytest systemone/tests -m "not slow"   # fast, no model needed
pytest systemone/tests                  # includes one end-to-end model test
```

## Install (Windows, RTX GPU)

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install transformers scikit-learn numpy scipy tqdm "mcp<2" packaging
pip install -e .   # installs the local gliclass fork (from repo root)
```

Then `python -m systemone.mcp_server` or import `systemone` anywhere.
Set `PYTHONIOENCODING=utf-8` on Windows consoles.
