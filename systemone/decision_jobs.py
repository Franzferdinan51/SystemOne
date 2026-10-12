"""Non-routing typed decision jobs.

State in, a batch of ``choice`` / ``score`` / ``noul`` questions out.
These are the jobs people run on decision models besides picking a model
or an effort tier:

- support / ops triage (queue, urgency, human handoff) in one batch
- a guard over a closed action or tool set that can abstain or withhold
  before any side effect
- citation grounding (which passage, how strong, is the claim grounded)

Nothing here calls ``POST /v1/systemone/route`` or names a model tier.
Engines judge the questions this module builds; this module only builds
them, with :func:`systemone.patterns.make_questions`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from .patterns import make_questions, with_abstain

# Support / ops triage. One batch covers queue, urgency, and handoff.
TRIAGE_QUEUES = ["billing", "technical", "account", "abuse"]
TRIAGE_URGENCY = ["low", "medium", "high", "critical"]

# Closed tool set for the pre-side-effect guard. ``with_abstain`` appends
# ``none`` so the judge can refuse to act instead of picking a bad tool.
GUARD_ACTIONS = ["send_reply", "issue_refund", "close_ticket", "escalate"]
GUARD_RISK = ["low", "moderate", "high", "blocking"]

# Citation grounding. Passage ids are the closed choice; ``none`` means
# no supplied passage supports the claim.
GROUNDING_PASSAGES = ["passage_a", "passage_b", "passage_c"]
GROUNDING_SUPPORT = ["contradicted", "unsupported", "partial", "supported"]

SUPPORT_TRIAGE_STATE = (
    "Customer ticket: Payment service returning 500s for 12 minutes. "
    "Error rate 40% and climbing. On-call paged 5 minutes ago, no ack. "
    "Third incident this week. Checkout is blocked for all regions."
)

ACTION_GUARD_STATE = (
    "Proposed tool call, not yet executed: issue_refund for order 88421, "
    "amount $240, reason 'duplicate charge'. The customer mentions a "
    "duplicate, but the ledger shows a single capture. Policy allows "
    "refunds under $50 without review; this amount is over that cap."
)

CITATION_GROUNDING_STATE = (
    "Claim: The refund window is 30 days from delivery.\n"
    "Passage A: Orders may be returned within 30 days of the delivery date.\n"
    "Passage B: Store credit expires 12 months after issue.\n"
    "Passage C: Shipping delays do not extend the statutory warranty."
)


# Each spec is data. ``build_job`` is the only constructor, and it calls
# the shipped question builder. ``abstain_on`` names choice questions that
# gain an explicit refuse-to-act option.
JOBS: List[Dict[str, Any]] = [
    {
        "id": "support_triage",
        "state": SUPPORT_TRIAGE_STATE,
        "choices": {"queue": TRIAGE_QUEUES},
        "scores": {"urgency": TRIAGE_URGENCY},
        "nouls": {
            "needs_human": (
                "A human must handle this ticket before any automated "
                "reply or side effect."
            ),
        },
        "prompts": {
            "queue": "Which queue should take this ticket?",
            "urgency": "How urgent is this ticket?",
        },
        "abstain_on": (),
    },
    {
        "id": "action_guard",
        "state": ACTION_GUARD_STATE,
        "choices": {"action": GUARD_ACTIONS},
        "scores": {"risk": GUARD_RISK},
        "nouls": {
            "withhold": (
                "Withhold every side effect until a person confirms "
                "the action."
            ),
        },
        "prompts": {
            "action": (
                "Which action is allowed before any side effect? "
                "Select none if the right action is unclear."
            ),
            "risk": "How risky is executing a side effect right now?",
        },
        "abstain_on": ("action",),
    },
    {
        "id": "citation_grounding",
        "state": CITATION_GROUNDING_STATE,
        "choices": {"supporting_passage": GROUNDING_PASSAGES},
        "scores": {"support": GROUNDING_SUPPORT},
        "nouls": {
            "grounded": "The claim is grounded in one of the supplied passages.",
        },
        "prompts": {
            "supporting_passage": (
                "Which passage grounds the claim? "
                "Select none if no passage supports it."
            ),
            "support": "How strongly do the passages support the claim?",
        },
        "abstain_on": ("supporting_passage",),
    },
]


def choice_options(spec: Mapping[str, Any]) -> Dict[str, List[str]]:
    """Choice lists for one job, with abstain applied where the spec asks."""
    abstain_on = set(spec.get("abstain_on") or ())
    out: Dict[str, List[str]] = {}
    for name, options in dict(spec["choices"]).items():
        labels = [str(option) for option in options]
        out[name] = with_abstain(labels) if name in abstain_on else labels
    return out


def build_questions(spec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Typed questions for one job, built by :func:`make_questions`."""
    questions = make_questions(
        choices=choice_options(spec),
        scores={name: list(levels) for name, levels in dict(spec["scores"]).items()},
        nouls={name: str(text) for name, text in dict(spec["nouls"]).items()},
    )
    prompts = dict(spec.get("prompts") or {})
    for question in questions:
        prompt = prompts.get(question["name"])
        if prompt:
            question["prompt"] = prompt
    return questions


def build_job(
    spec: Mapping[str, Any],
    state: Optional[str] = None,
) -> Dict[str, Any]:
    """``{"state", "questions"}`` for a decide/batch call. Not a route body."""
    return {
        "state": spec["state"] if state is None else state,
        "questions": build_questions(spec),
    }


def job_by_id(job_id: str) -> Dict[str, Any]:
    """Return the spec named ``job_id``."""
    for spec in JOBS:
        if spec["id"] == job_id:
            return spec
    known = ", ".join(spec["id"] for spec in JOBS)
    raise KeyError(f"unknown decision job {job_id!r}; known: {known}")


def support_triage(state: Optional[str] = None) -> Dict[str, Any]:
    """Ops triage: queue, urgency, and whether a human must take it."""
    return build_job(job_by_id("support_triage"), state)


def action_guard(
    state: Optional[str] = None,
    actions: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Closed action set. Abstain (``none``) withholds the side effect.

    ``actions`` replaces the default tool set. The abstain option is always
    appended. Callers must not execute a tool when the choice is ``none``
    or when ``withhold`` is answered true.
    """
    spec = job_by_id("action_guard")
    if actions is not None:
        spec = {**spec, "choices": {"action": [str(action) for action in actions]}}
    return build_job(spec, state)


def citation_grounding(
    state: Optional[str] = None,
    passage_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Which passage grounds a claim, how strong that support is, and whether it is grounded."""
    spec = job_by_id("citation_grounding")
    if passage_ids is not None:
        spec = {
            **spec,
            "choices": {"supporting_passage": [str(pid) for pid in passage_ids]},
        }
    return build_job(spec, state)
