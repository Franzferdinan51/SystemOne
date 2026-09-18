"""Domain tuning helper: fine-tune GLiClass on your own decision data.

Thin wrapper around the repo's train.py. Two workflows:

1. Full fine-tune of a small GLiClass checkpoint on labeled decisions:
       from systemone.tune import make_training_json, tune
       make_training_json(records, "data/my_decisions.json")
       tune("knowledgator/gliclass-edge-v3.0", "data/my_decisions.json")

2. LoRA adapters (parameter-efficient, keeps base weights frozen) are
   supported by the repo's training stack; pass lora=True to tune() and
   the resulting adapter can be selected per-request via the pipeline's
   adapter_ids argument.

Record format (what YOU write):
    {"text": "...", "labels": ["a", "b", "c"], "true_labels": ["b"]}
    {"text": "...", "labels": {"group": ["x", "y"]}, "true_labels": ["group.y"]}

See examples/sample_decision_data.json for routing + guardrail examples.
"""

from __future__ import annotations

import json
import os
from typing Dict, List, Sequence

from .distill import to_gliclass_json, train_student


def make_training_json(records: Sequence[Dict], path: str) -> str:
    """Convert simple decision records to train.py's JSON format.

    Hierarchical labels may use dot notation ("group.y") or nested dicts;
    they are flattened to dot notation here.
    """
    flat: List[Dict] = []
    for r in records:
        labels = r["labels"]
        if isinstance(labels, dict):
            labels = _flatten(labels)
        true = [ _flatten_label(t) for t in r["true_labels"]]
        flat.append({"text": r["text"], "labels": labels, "true_labels": true})
    return to_gliclass_json(flat, path)


def _flatten(d: Dict, prefix: str = "") -> List[str]:
    out: List[str] = []
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.extend(_flatten(v, key))
        elif isinstance(v, list):
            out.extend(f"{key}.{x}" if not str(x).startswith(key + ".") else str(x) for x in v)
        else:
            out.append(key)
    return out


def _flatten_label(t: str) -> str:
    return t  # already dot-notation by convention


def tune(
    model_name: str,
    data_path: str,
    save_path: str = "models/tuned",
    problem_type: str = "single_label_classification",
    num_epochs: int = 3,
    batch_size: int = 16,
    encoder_lr: float = 1e-5,
    others_lr: float = 3e-5,
    lora: bool = False,
    dry_run: bool = False,
) -> List[str]:
    """Fine-tune a GLiClass checkpoint on your decision data.

    Args:
        lora: if True, passes LoRA flags through to train.py when the
              repo's training stack supports them (see train.py --help).
              Falls back to full fine-tune otherwise.
    Returns the exact train.py command (runs it unless dry_run).
    """
    extra = [
        "--encoder_lr", str(encoder_lr),
        "--others_lr", str(others_lr),
    ]
    if lora:
        extra += ["--use_lora", "True"]
    return train_student(
        data_path,
        model_name=model_name,
        save_path=save_path,
        problem_type=problem_type,
        num_epochs=num_epochs,
        batch_size=batch_size,
        extra_args=extra,
        dry_run=dry_run,
    )
