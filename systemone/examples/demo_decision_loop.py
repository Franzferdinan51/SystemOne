"""Fast decision-loop demo: the SGLang Pokemon pattern, distilled.

Each tick is ONE batched decisions call carrying several questions at once
(action choice + stuck? gate + progress score), answered in a single pass —
the loop shape SGLang's /v1/decisions was built for:

    capture state -> build questions -> ONE decisions call -> act ->
    compress turn into memory -> repeat until done / step budget

Patterns ported from the agent-loop research (see README "SGLang interop"):
  * keep the prompt prefix byte-identical across ticks so SGLang's
    RadixAttention prefix cache reuses the KV cache (shared system prompt
    first, changing state second, fresh observation last);
  * only the latest observation is carried raw — older turns are compressed
    into "I saw / I thought / I did" one-liners (vision-only Pokemon agents
    do exactly this; screenshots cost ~6x tokens/step);
  * StallGuard trips when consecutive ticks make no progress;
  * gate on label_mass (SGLang) / confidence: low -> fall back to WAIT
    instead of acting on a guess.

The world here is a tiny mock grid (treasure hunt) so the demo runs fully
offline with a scripted judge. Point --engine at a real backend to watch
the same loop drive real decisions:

    python examples/demo_decision_loop.py --engine stub    # offline (default)
    python examples/demo_decision_loop.py --engine local   # GLiClass
    SGLANG_BASE_URL=... python examples/demo_decision_loop.py --engine sglang

Nothing is at stake: the "executor" just moves a cursor on a grid.
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from systemone import StallGuard, make_questions  # noqa: E402

# -- mock world: 6x6 grid, treasure at (5, 4), lava pits --------------------
W, H = 6, 6
TREASURE = (5, 4)
PITS = {(2, 2), (3, 4), (1, 4)}
ACTIONS = ["up", "down", "left", "right", "wait"]
DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0),
         "wait": (0, 0)}

SYSTEM_PREFIX = (
    "You are a game-playing agent. Each turn you see the game state and "
    "choose ONE action. Rules: reach the treasure (*). Avoid lava pits (X). "
    "Do not walk into walls. If stuck, say so."
)


def render(state) -> str:
    """Text viewport of the grid around the player."""
    px, py = state["player"]
    rows = []
    for y in range(H):
        row = ""
        for x in range(W):
            if (x, y) == (px, py):
                row += "@"
            elif (x, y) == TREASURE:
                row += "*"
            elif (x, y) in PITS:
                row += "X"
            else:
                row += "."
        rows.append(row)
    return "\n".join(rows)


def build_state_text(memory_lines, obs_text, tick) -> str:
    """Byte-identical prefix first (cache-friendly), fresh state last."""
    mem = "\n".join(memory_lines[-6:]) if memory_lines else "(no history yet)"
    return (
        f"{SYSTEM_PREFIX}\n\n"
        f"Memory (compressed turns):\n{mem}\n\n"
        f"Tick {tick} — current observation:\n{obs_text}"
    )


def build_questions() -> list:
    return make_questions(
        choices={"action": ACTIONS},
        scores={"progress": ["0", "1", "2", "3", "4"]},
        nouls={"stuck": "Is the agent stuck (repeating positions with no progress)?"},
    )


# -- judges -----------------------------------------------------------------
def stub_judge(state, questions):
    """Deterministic offline judge: greedy toward the treasure, avoids pits.

    Returns api-shaped answers so the loop below is backend-agnostic.
    """
    px, py = state["player"]
    tx, ty = TREASURE
    # greedy: reduce manhattan distance, avoid pits/walls
    best, best_d = "wait", abs(px - tx) + abs(py - ty)
    for a in ["right", "down", "left", "up"]:
        dx, dy = DELTA[a]
        nx, ny = px + dx, py + dy
        if not (0 <= nx < W and 0 <= ny < H) or (nx, ny) in PITS:
            continue
        d = abs(nx - tx) + abs(ny - ty)
        if d < best_d:
            best, best_d = a, d
    n = len(ACTIONS)
    probs = {a: (0.7 if a == best else 0.3 / (n - 1)) for a in ACTIONS}
    stuck = len(state["trail"]) >= 4 and len(set(state["trail"][-4:])) == 1
    return {
        "action": {"type": "choice", "choice": best, "probabilities": probs,
                   "confidence": 0.7, "label_mass": None},
        "progress": {"type": "score", "level": str(min(4, state["tick"] // 4)),
                     "distribution": {"0": 0.1, "1": 0.15, "2": 0.2,
                                      "3": 0.25, "4": 0.3},
                     "confidence": 0.3, "label_mass": None},
        "stuck": {"type": "noul", "probability": 0.9 if stuck else 0.05,
                  "answer": stuck, "confidence": 0.9 if stuck else 0.95,
                  "label_mass": None},
        "_meta": {"backend": "stub", "model": "stub", "latency_ms": 0.1},
    }


def make_judge(engine_name):
    if engine_name == "stub":
        return lambda state_text, questions, **kw: stub_judge(kw["world"], questions)
    if engine_name == "local":
        from systemone import SystemOne
        eng = SystemOne()
        return lambda state_text, questions, **kw: eng.systemone(state_text, questions)
    if engine_name == "sglang":
        from systemone import SGLangBackend
        eng = SGLangBackend()
        if not eng.health():
            raise SystemExit(f"SGLang not reachable at {eng.base_url}")
        return lambda state_text, questions, **kw: eng.systemone(state_text, questions)
    raise SystemExit(f"unknown engine: {engine_name}")


# -- the loop ----------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="stub", choices=["stub", "local", "sglang"])
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    random.seed(args.seed)

    judge = make_judge(args.engine)
    world = {"player": (0, 0), "tick": 0, "trail": []}
    memory: list[str] = []          # compressed "I saw / thought / did" lines
    guard = StallGuard(max_stalls=3)
    prev_dist = None

    for tick in range(args.budget):
        world["tick"] = tick
        obs = render(world)
        state_text = build_state_text(memory, obs, tick)
        answers = judge(state_text, build_questions(), world=world)
        meta = answers.get("_meta", {})

        action = answers["action"]["choice"]
        conf = answers["action"]["confidence"]
        mass = answers["action"].get("label_mass")
        # Uncertainty gate: low label_mass (SGLang) or low confidence -> wait.
        if (mass is not None and mass < 0.5) or conf < 0.35:
            action = "wait"

        px, py = world["player"]
        dx, dy = DELTA[action]
        nx, ny = px + dx, py + dy
        if 0 <= nx < W and 0 <= ny < H and (nx, ny) not in PITS:
            world["player"] = (nx, ny)
        world["trail"].append(world["player"])

        dist = abs(world["player"][0] - TREASURE[0]) + abs(world["player"][1] - TREASURE[1])
        progressed = prev_dist is None or dist < prev_dist
        prev_dist = dist
        status = guard.observe(progressed)

        memory.append(
            f"tick {tick}: I saw player at {world['player']}; "
            f"I thought action={action} (conf {conf:.2f}); "
            f"I did move to {world['player']}."
        )
        print(f"tick {tick:2d} action={action:5s} conf={conf:.2f} "
              f"pos={world['player']} dist={dist} "
              f"[{meta.get('backend')}/{meta.get('latency_ms')}ms]")

        if world["player"] == TREASURE:
            print(f"\nTreasure reached in {tick + 1} ticks.")
            return
        if status == "stalled" or answers["stuck"]["answer"]:
            print(f"\nStalled at tick {tick} — stopping instead of spinning.")
            return
    print(f"\nStep budget ({args.budget}) exhausted.")


if __name__ == "__main__":
    main()
