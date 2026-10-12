# systemone-bridge

Bridges Hermes to SystemOne's DecisionLoop (`~/SystemOne`). More than a
safety gate: it exposes SystemOne's Envs, batched decisions, and the full
See → Decide → Act loop as first-class Hermes capabilities.

## What it does

| Piece | How |
|---|---|
| `pre_tool_call` hook | Confidence-gates high-risk calls (deletes, dangerous shell like `rm -rf`/`sudo`/`curl\|sh`, writes to sensitive paths). Blocks below 0.70 confidence; raises with the reason so Hermes sees it. |
| `post_tool_call` hook | Logs every tool call (arg **shapes** only — names + sizes, never values) to `calibration.jsonl` for later judge calibration. |
| `systemone_env` tool | Run one action string in `filesystem` / `shell` / `webfetch` Envs. Env state persists per `(env, start_dir)` within the session. |
| `systemone_decide` tool | Answer multiple `choice` / `noul` questions in **one** batched call (the `/v1/decisions` pattern). Returns `{name: {choice, confidence}}`. |
| `systemone_run` tool | Full See → Decide → Act loop: you give a task + candidate action strings, the loop picks one per tick with confidence gating, stall detection, and turn memory. Returns the step transcript and outcome (`done` / `stalled` / `budget`). |
| `/systemone-bridge` command | `status` · `test <action…>` · `decisions` · `decide <json>` |

## Examples

```
/systemone-bridge test rm -rf /tmp/cache
# approved=False confidence=0.05 ...

/systemone-bridge decide [{"name":"queue","type":"choice","options":["billing","technical","abuse"]}]
```

Tool call `systemone_decide`:
```json
{
  "questions": [
    {"name": "urgent", "type": "noul"},
    {"name": "queue", "type": "choice", "options": ["billing", "technical", "abuse"]}
  ],
  "context": "Customer ticket: charged twice for the Pro plan, wants a refund."
}
```

Tool call `systemone_run`:
```json
{
  "task": "Find which file in /tmp/logs is largest",
  "env": "filesystem",
  "candidates": ["ls /tmp/logs", "pwd", "done"],
  "max_ticks": 6,
  "start_dir": "/tmp"
}
```

## v1 notes

- Judges are **heuristic** (risk-pattern + keyword-overlap scoring, stdlib
  only) — no model dependency. A trained decision model (e.g. a JEV-style
  adapter) can replace `_gate_judge` / `_choice_judge` in `bridge.py`
  without changing the interface.
- **Fail-open everywhere**: if SystemOne can't be imported or anything
  raises, actions are allowed through and the reason is logged.
- Gate decisions append to `decisions.log`; calibration data to
  `calibration.jsonl` (same directory).
- The task runner needs **candidate actions from Hermes** (the LLM brain);
  SystemOne provides the picking, gating, stall detection, and memory.

## Files

- `plugin.yaml` — metadata, hooks
- `__init__.py` — hook handlers, tool handlers, slash command, `register()`
- `bridge.py` — `SystemOneBridge`: DecisionLoop wiring, Envs, judges, logging
- `README.md` — this file
