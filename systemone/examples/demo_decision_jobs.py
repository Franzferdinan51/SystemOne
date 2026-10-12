"""Print the non-routing decision jobs. No model and no shim.

Run:  python systemone/examples/demo_decision_jobs.py

Each payload is ``{"state", "questions"}`` for ``systemone(state, questions)``
or ``POST /v1/systemone/decide`` / ``/batch``. None of them is a route body.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from systemone.decision_jobs import (  # noqa: E402
    action_guard,
    citation_grounding,
    support_triage,
)


def main() -> None:
    jobs = {
        "support_triage": support_triage(),
        "action_guard": action_guard(),
        "citation_grounding": citation_grounding(),
    }
    print(json.dumps(jobs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
