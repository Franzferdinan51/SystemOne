"""Non-routing decision jobs pass the shipped question builder and validators.

No model and no shim. The jobs are state plus choice / score / noul
questions. Routing payloads are a different shape and these are not those.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from systemone.cli import build_parser
from systemone.decision_jobs import (
    JOBS,
    action_guard,
    build_job,
    citation_grounding,
    support_triage,
)
from systemone.patterns import (
    ABSTAIN_LABEL,
    build_decision_prompts,
    make_questions,
    validate_choice,
    validate_distribution,
    with_abstain,
)


def _one_hot(labels):
    """A legal distribution: finite, sums to 1, first label holds the max."""
    return {label: (1.0 if i == 0 else 0.0) for i, label in enumerate(labels)}


def _drive_validators(question, labels):
    """Run the shipped response-contract checks on this question's labels."""
    probs = _one_hot(labels)
    if question["type"] == "choice":
        validate_choice(
            {"choice": labels[0], "probabilities": probs, "confidence": 1.0},
            labels,
        )
    else:
        winner = labels[0]
        validate_distribution(probs, labels, winner)


def _assert_not_route(payload):
    assert set(payload) == {"state", "questions"}
    assert payload["state"]
    assert payload["questions"]
    encoded = json.dumps(payload)
    assert "/v1/systemone/route" not in encoded
    for key in ("task", "cost_bias", "tiers", "route", "tier", "model_id"):
        assert key not in payload
    for question in payload["questions"]:
        assert question["type"] in {"choice", "score", "noul"}
        assert "tier" not in question
        assert "model_id" not in question


def test_three_jobs_use_builder_and_pass_validators():
    assert [spec["id"] for spec in JOBS] == [
        "support_triage",
        "action_guard",
        "citation_grounding",
    ]
    payloads = {
        "support_triage": support_triage(),
        "action_guard": action_guard(),
        "citation_grounding": citation_grounding(),
    }
    for spec in JOBS:
        # Drive make_questions itself, then require the shipped job to match it.
        choices = {
            name: (
                with_abstain(list(options))
                if name in set(spec["abstain_on"])
                else list(options)
            )
            for name, options in spec["choices"].items()
        }
        built = make_questions(
            choices=choices,
            scores={name: list(levels) for name, levels in spec["scores"].items()},
            nouls=dict(spec["nouls"]),
        )
        payload = payloads[spec["id"]]
        assert build_job(spec)["questions"][0]["type"] == built[0]["type"]
        assert [q["name"] for q in payload["questions"]] == [q["name"] for q in built]
        for question, bare in zip(payload["questions"], built):
            assert question["type"] == bare["type"]
            if question["type"] == "choice":
                assert question["options"] == bare["options"]
            elif question["type"] == "score":
                assert question["levels"] == bare["levels"]
            else:
                assert question["statement"] == bare["statement"]
        assert {q["type"] for q in payload["questions"]} == {"choice", "score", "noul"}
        _assert_not_route(payload)
        prompts, label_lists = build_decision_prompts(
            payload["state"], payload["questions"]
        )
        assert len(prompts) == len(payload["questions"]) == len(label_lists)
        for question, labels, prompt in zip(
            payload["questions"], label_lists, prompts
        ):
            assert labels
            assert payload["state"][:40] in prompt
            _drive_validators(question, labels)


def test_triage_is_one_batch_of_choice_score_and_noul():
    payload = support_triage()
    kinds = [q["type"] for q in payload["questions"]]
    assert kinds == ["choice", "score", "noul"]
    queue = payload["questions"][0]
    assert queue["name"] == "queue"
    assert queue["options"] == ["billing", "technical", "account", "abuse"]
    assert payload["questions"][1]["levels"] == ["low", "medium", "high", "critical"]
    assert "human" in payload["questions"][2]["statement"].lower()


def test_action_guard_can_abstain_or_withhold():
    payload = action_guard()
    action = next(q for q in payload["questions"] if q["name"] == "action")
    withhold = next(q for q in payload["questions"] if q["name"] == "withhold")
    assert action["type"] == "choice"
    assert action["options"][-1] == ABSTAIN_LABEL
    assert action["options"][:-1] == [
        "send_reply",
        "issue_refund",
        "close_ticket",
        "escalate",
    ]
    assert withhold["type"] == "noul"
    assert "withhold" in withhold["statement"].lower()
    custom = action_guard(actions=["delete_file", "send_email"])
    custom_action = next(q for q in custom["questions"] if q["name"] == "action")
    assert custom_action["options"] == ["delete_file", "send_email", ABSTAIN_LABEL]


def test_citation_grounding_names_passages_and_can_abstain():
    payload = citation_grounding()
    passage = next(q for q in payload["questions"] if q["name"] == "supporting_passage")
    support = next(q for q in payload["questions"] if q["name"] == "support")
    grounded = next(q for q in payload["questions"] if q["name"] == "grounded")
    assert passage["options"] == [
        "passage_a",
        "passage_b",
        "passage_c",
        ABSTAIN_LABEL,
    ]
    assert support["type"] == "score"
    assert "supported" in support["levels"]
    assert grounded["type"] == "noul"
    assert "grounded" in grounded["statement"].lower()
    swapped = citation_grounding(passage_ids=["p1", "p2"])
    opts = next(
        q["options"] for q in swapped["questions"] if q["name"] == "supporting_passage"
    )
    assert opts == ["p1", "p2", ABSTAIN_LABEL]


def test_cli_help_leads_with_a_typed_decision():
    text = build_parser().format_help()
    lines = text.splitlines()
    assert lines[0].startswith("usage:")
    assert "decide" in lines[0]
    assert "route" not in lines[0]
    description = next(
        line for line in lines if line.startswith("Open typed-decision layer")
    )
    assert description.lower().startswith("open typed-decision layer")
    assert "choice" in description and "score" in description and "noul" in description
    # The first named command in the help listing is the typed decision.
    command_helps = [
        line.strip()
        for line in lines
        if line.strip().startswith("decide ") or line.strip().startswith("route ")
    ]
    assert command_helps
    assert command_helps[0].startswith("decide ")
