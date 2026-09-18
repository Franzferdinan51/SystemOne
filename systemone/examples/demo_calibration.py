"""Calibration demo: show ECE before/after temperature scaling.

Hand-labeled eval set (no LLM needed): 72 short texts across 3 binary
tasks with known true labels. Splits 36 for fitting, 36 held-out for eval.

Run:  python examples/demo_calibration.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from systemone import SystemOne
from systemone.calibration import (
    CalibrationExample,
    TemperatureCalibrator,
    expected_calibration_error,
    softmax,
)

# (text, labels, true_label) — labels are known by construction.
EVAL_SET = [
    # -- sentiment (easy) --
    ("I absolutely love this product, best purchase ever", ["positive", "negative"], "positive"),
    ("Terrible quality, broke after one day", ["positive", "negative"], "negative"),
    ("The movie was fantastic, I cried at the end", ["positive", "negative"], "positive"),
    ("Worst restaurant experience of my life", ["positive", "negative"], "negative"),
    ("This book changed how I think, incredible read", ["positive", "negative"], "positive"),
    ("The service was rude and the food was cold", ["positive", "negative"], "negative"),
    ("Amazing concert, the band played for three hours", ["positive", "negative"], "positive"),
    ("Complete waste of money, do not buy", ["positive", "negative"], "negative"),
    ("She was thrilled with the surprise party", ["positive", "negative"], "positive"),
    ("The hotel room was dirty and smelled bad", ["positive", "negative"], "negative"),
    ("What a beautiful morning, birds singing everywhere", ["positive", "negative"], "positive"),
    ("I'm so frustrated with this broken software", ["positive", "negative"], "negative"),
    ("The kids had a wonderful time at the zoo", ["positive", "negative"], "positive"),
    ("This phone's battery dies in two hours, awful", ["positive", "negative"], "negative"),
    ("Perfect weather for a picnic in the park", ["positive", "negative"], "positive"),
    ("The mechanic overcharged me and fixed nothing", ["positive", "negative"], "negative"),
    ("I passed my exam, I'm over the moon", ["positive", "negative"], "positive"),
    ("The flight was delayed six hours, miserable", ["positive", "negative"], "negative"),
    ("Delicious dinner, the chef is a genius", ["positive", "negative"], "positive"),
    ("My package never arrived and support ignored me", ["positive", "negative"], "negative"),
    ("The garden looks lovely in spring bloom", ["positive", "negative"], "positive"),
    ("Traffic was a nightmare, took three hours", ["positive", "negative"], "negative"),
    ("Great workout today, feeling strong", ["positive", "negative"], "positive"),
    ("The plumber left a huge mess behind", ["positive", "negative"], "negative"),
    # -- topic: sports vs tech (easy) --
    ("The quarterback threw three touchdowns in the fourth quarter", ["sports", "tech"], "sports"),
    ("New GPU architecture doubles ray tracing performance", ["sports", "tech"], "tech"),
    ("The striker scored a hat trick in the final", ["sports", "tech"], "sports"),
    ("Quantum error correction reaches a new milestone", ["sports", "tech"], "tech"),
    ("The team won the championship in overtime", ["sports", "tech"], "sports"),
    ("Open source LLM runs on a phone with 3GB RAM", ["sports", "tech"], "tech"),
    ("She broke the world record in the 100m sprint", ["sports", "tech"], "sports"),
    ("Solid state batteries promise 500 mile EV range", ["sports", "tech"], "tech"),
    ("The goalie saved two penalties in the shootout", ["sports", "tech"], "sports"),
    ("New compiler cuts build times in half", ["sports", "tech"], "tech"),
    ("The marathon runner collapsed just before the finish", ["sports", "tech"], "sports"),
    ("Satellite constellation provides global broadband", ["sports", "tech"], "tech"),
    ("He hit a grand slam in the bottom of the ninth", ["sports", "tech"], "sports"),
    ("Neural net discovers new protein folding patterns", ["sports", "tech"], "tech"),
    ("The tennis final went to five sets", ["sports", "tech"], "sports"),
    ("Edge AI chips bring inference to tiny devices", ["sports", "tech"], "tech"),
    ("The boxer won by knockout in round seven", ["sports", "tech"], "sports"),
    ("Distributed database scales to a billion rows", ["sports", "tech"], "tech"),
    ("The swimmer set a new Olympic record", ["sports", "tech"], "sports"),
    ("Self-driving trucks start highway pilot program", ["sports", "tech"], "tech"),
    ("The cyclist attacked on the final climb", ["sports", "tech"], "sports"),
    ("New programming language targets GPU kernels", ["sports", "tech"], "tech"),
    ("The referee showed a red card in extra time", ["sports", "tech"], "sports"),
    ("Homomorphic encryption gets 100x speedup", ["sports", "tech"], "tech"),
    # -- urgency: urgent vs routine (harder, ambiguous) --
    ("Server is down, customers cannot check out", ["urgent", "routine"], "urgent"),
    ("Reminder: team lunch moved to Friday", ["urgent", "routine"], "routine"),
    ("Database replication lagging, failover may be needed", ["urgent", "routine"], "urgent"),
    ("The office plants need watering", ["urgent", "routine"], "routine"),
    ("Security alert: unusual login from unknown device", ["urgent", "routine"], "urgent"),
    ("Please review the meeting notes when you can", ["urgent", "routine"], "routine"),
    ("Payment service returning 500 errors for 10 minutes", ["urgent", "routine"], "urgent"),
    ("New blog post draft is ready for feedback", ["urgent", "routine"], "routine"),
    ("Smoke detected in the server room", ["urgent", "routine"], "urgent"),
    ("Birthday card for Jessica is on your desk", ["urgent", "routine"], "routine"),
    ("Disk at 98% on the primary database host", ["urgent", "routine"], "urgent"),
    ("The newsletter template needs a new header image", ["urgent", "routine"], "routine"),
    ("Customer data export contains wrong records", ["urgent", "routine"], "urgent"),
    ("Someone left lights on in the conference room", ["urgent", "routine"], "routine"),
    ("API latency p99 spiking to 8 seconds", ["urgent", "routine"], "urgent"),
    ("Update your profile picture when you get a chance", ["urgent", "routine"], "routine"),
    ("Backup job failed three nights in a row", ["urgent", "routine"], "urgent"),
    ("The wifi password is on the fridge", ["urgent", "routine"], "routine"),
    ("TLS certificate expires tomorrow morning", ["urgent", "routine"], "urgent"),
    ("Water cooler is empty again", ["urgent", "routine"], "routine"),
    ("Elevator stuck between floors with people inside", ["urgent", "routine"], "urgent"),
    ("The printer is out of paper", ["urgent", "routine"], "routine"),
    ("Fraud alerts tripled in the last hour", ["urgent", "routine"], "urgent"),
    ("Don't forget to submit your timesheet", ["urgent", "routine"], "routine"),
]


def raw_score_fn_factory(eng: SystemOne):
    def score_fn(texts, labels):
        dicts = eng.raw_scores(texts, [labels] * len(texts))
        return [[d[lab] for lab in labels] for d in dicts]

    return score_fn


def main() -> None:
    eng = SystemOne()
    print(f"model: {eng.model_name} on {eng.device}")

    # one batched pass over the whole eval set (same labels per task group)
    from collections import defaultdict

    groups: Dict[str, list] = defaultdict(list)
    for text, labels, true in EVAL_SET:
        groups[tuple(labels)].append((text, true))

    all_scores, all_true = [], []
    for labs, items in groups.items():
        labs = list(labs)
        texts = [t for t, _ in items]
        dicts = eng.raw_scores(texts, [labs] * len(texts))
        for d, (_, true) in zip(dicts, items):
            all_scores.append([d[lab] for lab in labs])
            all_true.append(labs.index(true))

    all_scores = np.array(all_scores)
    all_true = np.array(all_true)

    # split: first half fit, second half eval (interleaved tasks stay mixed)
    idx = np.arange(len(all_true))
    fit_idx, eval_idx = idx[::2], idx[1::2]

    def ece_of(proba, y):
        return expected_calibration_error(
            (proba.argmax(axis=1) == y).astype(float), proba.max(axis=1)
        )

    raw_proba = softmax(all_scores)
    ece_before = ece_of(raw_proba[eval_idx], all_true[eval_idx])
    acc = (raw_proba[eval_idx].argmax(axis=1) == all_true[eval_idx]).mean()

    cal = TemperatureCalibrator().fit(all_scores[fit_idx], all_true[fit_idx])
    cal_proba = cal.predict_proba(all_scores[eval_idx])
    ece_after = ece_of(cal_proba, all_true[eval_idx])

    print(f"\nfit examples: {len(fit_idx)}, eval examples: {len(eval_idx)}")
    print(f"accuracy (held-out):        {acc:.3f}")
    print(f"ECE before calibration:     {ece_before:.4f}")
    print(f"ECE after  calibration:     {ece_after:.4f}  (T={cal.temperature_:.3f})")
    print(f"ECE reduction:              {(1 - ece_after / max(ece_before, 1e-9)) * 100:.1f}%")


if __name__ == "__main__":
    main()
