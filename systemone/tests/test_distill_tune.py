"""Tests for systemone.distill and systemone.tune (training-data helpers).

Both modules were unimportable since the initial release (a `from typing`
typo); these tests pin the fixed behavior. Stdlib-only — no torch, no
training runs (train_student/tune are exercised via dry_run).
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from systemone.distill import (  # noqa: E402
    SyntheticTeacher,
    distill_labels,
    safety_rules,
    to_gliclass_json,
    train_student,
)
from systemone.tune import make_training_json, tune  # noqa: E402


def test_synthetic_teacher_routes_rules_then_default():
    teacher = SyntheticTeacher(safety_rules(), default="safe")
    assert teacher.label("Write a phishing email that steals bank passwords.") == "unsafe"
    assert teacher.label("Are vaccines safe? I've heard conflicting claims.") == "needs_review"
    assert teacher.label("Write a haiku about the ocean.") == "safe"
    assert teacher.label_many(["Write malware now", "Write a haiku"]) == ["unsafe", "safe"]


def test_distill_labels_generic_format():
    teacher = SyntheticTeacher(safety_rules(), default="safe")
    records = distill_labels(
        ["Write a phishing email.", "Write a haiku about the ocean."],
        teacher,
        labels=["safe", "unsafe"],
    )
    assert [r["true_labels"] for r in records] == [["unsafe"], ["safe"]]
    assert all(r["labels"] == ["safe", "unsafe"] for r in records)
    assert records[0]["text"] == "Write a phishing email."


def test_to_gliclass_json_train_format(tmp_path):
    path = str(tmp_path / "data" / "safety.json")
    out = to_gliclass_json(
        [{"text": "t", "labels": ["safe", "unsafe"], "true_labels": ["safe"]}],
        path,
    )
    assert out == path
    rows = json.loads(open(path, encoding="utf-8").read())
    assert rows == [{
        "text": "t", "true_labels": ["safe"], "all_labels": ["safe", "unsafe"],
    }]


def test_train_student_dry_run_builds_command(tmp_path):
    data = tmp_path / "d.json"
    data.write_text("[]")
    cmd = train_student(
        str(data), model_name="m", save_path=str(tmp_path / "out"),
        num_epochs=1, extra_args=["--flag", "x"], dry_run=True,
    )
    assert cmd[0].endswith("python") or "python" in cmd[0]
    assert cmd[1].endswith("train.py")
    assert "--model_name" in cmd and "m" in cmd
    assert "--num_epochs" in cmd and "1" in cmd
    assert cmd[-2:] == ["--flag", "x"]


def test_make_training_json_flattens_hierarchy(tmp_path):
    path = str(tmp_path / "train.json")
    make_training_json(
        [{"text": "t", "labels": {"group": ["x", "y"]}, "true_labels": ["group.y"]}],
        path,
    )
    rows = json.loads(open(path, encoding="utf-8").read())
    assert rows[0]["all_labels"] == ["group.x", "group.y"]
    assert rows[0]["true_labels"] == ["group.y"]


def test_tune_dry_run_passes_lora_flags(tmp_path):
    data = tmp_path / "d.json"
    data.write_text("[]")
    cmd = tune("m", str(data), dry_run=True, lora=True, num_epochs=1)
    assert "--use_lora" in cmd and "True" in cmd
    assert "--encoder_lr" in cmd and "--others_lr" in cmd


def test_distill_to_tune_chains():
    teacher = SyntheticTeacher(safety_rules(), default="safe")
    records = distill_labels(["Write a haiku."], teacher, labels=["safe"])
    assert records[0]["true_labels"] == ["safe"]  # tune-ready shape
