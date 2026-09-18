"""Headless 2048 decision-loop benchmark for the local SystemOne shim.

Replicates the *decision path* of Cua's jev-use 2048 demo: per step, build
a bounded set of action candidates (the four slides), ask the local
`/v1/systemone` endpoint to pick one ID, record wall-clock latency. Cost is
$0.00 by construction — local engine, no API key, no cloud.

HONEST LIMITS (read before quoting numbers):
- There is no browser on this VM, so nothing is actually *played*. The
  board states below are canned snapshots, not a live game; there is no
  2048 engine, no score, no win/lose. What this proves is the decision
  loop's latency and cost story: candidate construction -> local Jev-style
  decision -> verified choice, N times in a row.
- Cua's published figure (44.9s / ~$0.00108, demo by @injaneity) is a full
  game run against their stack. Our number is the headless decision-path
  equivalent on this hardware. Compare the *cost* claim ($0.00 local), not
  the wall clocks, unless you re-run both on the same box.

Run:
    python systemone/bench_2048.py [--steps 16]
    # env: SYSTEMONE_MODEL (default: smallest cached GLiClass),
    #      HF_HUB_OFFLINE=1 is set by default (no downloads).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.request

# Offline by default: the benchmark must never download.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from systemone.shim import serve  # noqa: E402

# Four canned board snapshots (4x4). Cycled through; the point is the
# decision loop, not the game.
BOARDS = [
    """2    4    8    16
32   64   128  256
512  1024 2    4
8    16   32   64""",
    """1024 512  256  128
64   32   16   8
4    2    4    8
16   32   64   128""",
    """2    2    4    8
16   32   64   128
256  512  1024 2
4    8    16   32""",
    """128  64   32   16
8    4    2    2
4    8    16   32
64   128  256  512""",
]

MOVES = {
    "up": "Slide all tiles up. Merges vertical pairs; keeps large tiles anchored at the top edge.",
    "down": "Slide all tiles down. Merges vertical pairs; keeps large tiles anchored at the bottom edge.",
    "left": "Slide all tiles left. Merges horizontal pairs; keeps large tiles anchored at the left edge.",
    "right": "Slide all tiles right. Merges horizontal pairs; keeps large tiles anchored at the right edge.",
}

GOAL = "Play 2048: merge tiles toward 2048. Prefer the slide that merges the most tile value while keeping the largest tile in a corner."
RULES = ["Pick exactly one of the four slides.", "Never repeat the previous slide twice in a row."]


def post_json(url: str, body: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def decide_move(base_url: str, board: str) -> tuple[str, float]:
    """One jev-use-style decision: candidates in, chosen move ID out."""
    body = {
        "model": "bench-2048",
        "state": f"2048 board state (4x4, rows top to bottom):\n{board}",
        "questions": {
            "move": {
                "type": "choice",
                "criteria": MOVES,
                "instructions": {"goal": GOAL, "rules": RULES},
            }
        },
    }
    t0 = time.perf_counter()
    payload = post_json(f"{base_url}/v1/systemone", body)
    ms = (time.perf_counter() - t0) * 1000.0
    move = payload["answers"]["move"]["choice"]
    assert move in MOVES, f"engine returned invalid move: {move!r}"
    return move, ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless 2048 decision-loop benchmark")
    parser.add_argument("--steps", type=int, default=16)
    args = parser.parse_args()

    server = serve(0)  # real local engine, ephemeral port
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    try:
        print(f"engine: {server.engine.model_name} (device {server.engine.device})")
        latencies: list[float] = []
        t_all = time.perf_counter()
        for i in range(args.steps):
            board = BOARDS[i % len(BOARDS)]
            move, ms = decide_move(base_url, board)
            latencies.append(ms)
            print(f"  step {i + 1:>2}/{args.steps}: move={move:<5} {ms:7.1f} ms")
        total_s = time.perf_counter() - t_all
    finally:
        server.shutdown()
        thread.join(timeout=5)

    mean_ms = statistics.mean(latencies)
    p50 = statistics.median(latencies)
    cost = 0.0  # local engine: no API key, no per-call charge, by construction

    print()
    print("=== 2048 decision-loop benchmark (headless, canned boards) ===")
    print(f"decisions        : {len(latencies)}")
    print(f"total wall       : {total_s:.1f} s")
    print(f"mean / p50       : {mean_ms:.1f} / {p50:.1f} ms per decision")
    print(f"cost             : ${cost:.2f} (local, no API key)")
    print()
    print("vs Cua jev-use published 2048 run: 44.9 s / ~$0.00108 API-equivalent")
    print(f"   SystemOne local (decision path): {total_s:.1f} s / ${cost:.2f}")
    print("   (their figure is a full game run; ours is the headless decision")
    print("    loop on this hardware — the comparable claim is the cost.)")


if __name__ == "__main__":
    main()
