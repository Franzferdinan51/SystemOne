"""systemone() demo: all three question types, one batched pass.

Also demonstrates the batching win: timing 1 question vs 6 questions —
latency should barely move.

Run:  python examples/demo_systemone.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from systemone import SystemOne, make_questions


def main() -> None:
    eng = SystemOne()
    print(f"model: {eng.model_name} on {eng.device}\n")

    state = (
        "Payment service returning 500 errors for 12 minutes. "
        "Error rate 40% and climbing. On-call engineer paged 5 minutes ago, "
        "no ack yet. This is the third incident this week."
    )

    questions = make_questions(
        choices={
            "team": ["backend", "frontend", "devops", "support"],
            "severity": ["sev1", "sev2", "sev3", "sev4"],
        },
        scores={
            "customer_impact": ["low", "medium", "high", "critical"],
        },
        nouls={
            "page_cto": "Should the CTO be woken up for this incident?",
            "postmortem": "Does this incident need a formal postmortem?",
            "all_clear": "Is it safe to tell customers the issue is resolved?",
        },
    )

    out = eng.systemone(state, questions)
    for q in questions:
        name = q["name"]
        print(f"--- {name} ({q['type']}) ---")
        a = out[name]
        if q["type"] == "choice":
            print(f"choice: {a['choice']}  (confidence {a['confidence']:.3f})")
            for k, v in sorted(a["probabilities"].items(), key=lambda kv: -kv[1]):
                print(f"    {k:12s} {v:.3f}")
        elif q["type"] == "score":
            print(f"level: {a['level']}  (confidence {a['confidence']:.3f})")
            for k, v in a["distribution"].items():
                print(f"    {k:12s} {v:.3f}")
        else:
            print(f"answer: {a['answer']}  P(true)={a['probability']:.3f}")
    print(f"\nmeta: {out['_meta']}")

    # batching win: 1 question vs 6 questions
    one = [{"name": "q", "type": "choice", "options": ["a", "b", "c"]}]
    t0 = time.perf_counter()
    eng.systemone(state, one)
    t1 = time.perf_counter()
    eng.systemone(state, questions)
    t2 = time.perf_counter()
    print(
        f"\n1 question: {(t1 - t0) * 1000:.0f} ms | "
        f"{len(questions)} questions: {(t2 - t1) * 1000:.0f} ms"
    )


if __name__ == "__main__":
    main()
