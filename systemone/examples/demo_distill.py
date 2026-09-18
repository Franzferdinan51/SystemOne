"""Distillation demo: content-safety classifier -> gliclass-edge.

Pipeline (no LLM needed for the demo):
  1. SyntheticTeacher labels 120 short texts (safe/unsafe/needs_review).
  2. Labels are written as GLiClass training JSON.
  3. (optional --train) fine-tune gliclass-edge via train.py, then eval.

For a real teacher, swap in HFTeacher or LMStudioTeacher in distill.py.

Run:
  python examples/demo_distill.py            # label + write data only
  python examples/demo_distill.py --train    # also fine-tune (~minutes on GPU)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from systemone.distill import (
    SyntheticTeacher,
    distill_labels,
    make_safety_demo_texts,
    safety_rules,
    to_gliclass_json,
    train_student,
)

LABELS = ["safe", "unsafe", "needs_review"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="fine-tune the student")
    ap.add_argument("--n", type=int, default=40, help="examples per class")
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args()

    texts = make_safety_demo_texts(n_per_class=args.n)
    teacher = SyntheticTeacher(safety_rules(), default="safe")
    records = distill_labels(texts, teacher, LABELS)
    print(f"labeled {len(records)} examples with SyntheticTeacher")

    from collections import Counter

    print("label distribution:", dict(Counter(r["true_labels"][0] for r in records)))

    data_path = os.path.join(os.path.dirname(__file__), "safety_distilled.json")
    to_gliclass_json(records, data_path)
    print(f"wrote {data_path}")

    if args.train:
        cmd = train_student(
            data_path,
            model_name="knowledgator/gliclass-edge-v3.0",
            save_path=os.path.join(os.path.dirname(__file__), "..", "models", "safety-edge"),
            problem_type="single_label_classification",
            num_epochs=args.epochs,
            batch_size=16,
            dry_run=True,
        )
        print("train command:")
        print("  " + " ".join(cmd))
        train_student(
            data_path,
            model_name="knowledgator/gliclass-edge-v3.0",
            save_path=os.path.join(os.path.dirname(__file__), "..", "models", "safety-edge"),
            problem_type="single_label_classification",
            num_epochs=args.epochs,
            batch_size=16,
        )
        print("done. student at models/safety-edge")


if __name__ == "__main__":
    main()
