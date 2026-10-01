"""Fast decision-loop demo: See > Decide > Act on a tiny grid world.

Each tick is ONE batched decisions call carrying several questions at once
(action choice + stuck? gate + progress score), answered in a single pass —
the loop shape SGLang's /v1/decisions was built for:

    SEE state -> DECIDE (one call) -> ACT -> compress turn -> repeat

The loop itself lives in systemone/loop.py (DecisionLoop); this demo only
provides the world (a mock treasure-hunt grid) and the judges, so it runs
fully offline with a scripted judge. Point --engine at a real backend to
watch the same loop drive real decisions:

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

from systemone import make_questions  # noqa: E402
from systemone.loop import ActResult, DecisionLoop, Observation  # noqa: E402

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


def render(player) -> str:
    """Text viewport of the grid around the player."""
    px, py = player
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


class GridEnv:
    """SEE the grid as text; ACT by moving the cursor."""

    def __init__(self, start=(0, 0)):
        self.player = start
        self.trail = []
        self.tick = 0
        self.prev_dist = None

    def observe(self):
        self._dist = abs(self.player[0] - TREASURE[0]) + abs(self.player[1] - TREASURE[1])
        return Observation(text=render(self.player))

    def act(self, action):
        dx, dy = DELTA.get(action, (0, 0))
        nx, ny = self.player[0] + dx, self.player[1] + dy
        if 0 <= nx < W and 0 <= ny < H and (nx, ny) not in PITS:
            self.player = (nx, ny)
        self.trail.append(self.player)
        self.tick += 1
        dist = abs(self.player[0] - TREASURE[0]) + abs(self.player[1] - TREASURE[1])
        progressed = self.prev_dist is None or dist < self.prev_dist
        self.prev_dist = dist
        return ActResult(progressed=progressed, done=self.player == TREASURE)


def build_questions() -> list:
    return make_questions(
        choices={"action": ACTIONS},
        scores={"progress": ["0", "1", "2", "3", "4"]},
        nouls={"stuck": "Is the agent stuck (repeating positions with no progress)?"},
    )


# -- judges -----------------------------------------------------------------
def stub_judge(state_text, questions, images, videos, *, env):
    """Deterministic offline judge: greedy toward the treasure, avoids pits.

    Returns api-shaped answers so the loop below is backend-agnostic.
    """
    px, py = env.player
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
    stuck = len(env.trail) >= 4 and len(set(env.trail[-4:])) == 1
    return {
        "action": {"type": "choice", "choice": best, "probabilities": probs,
                   "confidence": 0.7, "label_mass": None},
        "progress": {"type": "score", "level": str(min(4, env.tick // 4)),
                     "distribution": {"0": 0.1, "1": 0.15, "2": 0.2,
                                      "3": 0.25, "4": 0.3},
                     "confidence": 0.3, "label_mass": None},
        "stuck": {"type": "noul", "probability": 0.9 if stuck else 0.05,
                  "answer": stuck, "confidence": 0.9 if stuck else 0.95,
                  "label_mass": None},
        "_meta": {"backend": "stub", "model": "stub", "latency_ms": 0.1},
    }


def make_judge(engine_name, env):
    if engine_name == "stub":
        return lambda state_text, questions, images, videos: stub_judge(
            state_text, questions, images, videos, env=env)
    if engine_name == "local":
        from systemone import SystemOne
        eng = SystemOne()
        return lambda state_text, questions, images, videos: eng.systemone(
            state_text, questions, images=images or None, videos=videos or None)
    if engine_name == "sglang":
        from systemone import SGLangBackend
        eng = SGLangBackend()
        if not eng.health():
            raise SystemExit(f"SGLang not reachable at {eng.base_url}")
        return lambda state_text, questions, images, videos: eng.systemone(
            state_text, questions, images=images or None, videos=videos or None)
    raise SystemExit(f"unknown engine: {engine_name}")


# -- the loop ----------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="stub", choices=["stub", "local", "sglang"])
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    random.seed(args.seed)

    env = GridEnv()
    judge = make_judge(args.engine, env)
    loop = DecisionLoop(judge, budget=args.budget, system_prompt=SYSTEM_PREFIX)
    result = loop.run(env, build_questions())

    for step in result.steps:
        print(f"tick {step.tick:2d} action={step.action:5s} conf={step.confidence:.2f} "
              f"pos={env.trail[step.tick] if step.tick < len(env.trail) else env.player} "
              f"[{step.backend}/{step.latency_ms}ms]"
              + (" GATED" if step.gated else "")
              + (f" ESCALATED ({step.escalation})" if step.escalated else ""))

    if result.outcome == "done":
        print(f"\nTreasure reached in {result.n_ticks} ticks.")
    elif result.outcome == "stalled":
        print("\nStalled — stopping instead of spinning.")
    else:
        print(f"\nStep budget ({args.budget}) exhausted.")


if __name__ == "__main__":
    main()
