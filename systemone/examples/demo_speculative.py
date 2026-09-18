"""Speculative multi-head decisions in one batched forward pass.

Pattern ported from jev-ultrafast / mobile-jev: ask the operation AND every
plausible target in a single call, then keep only the target head that
matches the chosen operation. Unused heads cannot cause an action.

Usage:
    PYTHONPATH=. python systemone/examples/demo_speculative.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from systemone import SystemOne, with_abstain

eng = SystemOne(model_name=os.environ.get("SYSTEMONE_MODEL"))

out = eng.speculative_decide(
    state=(
        "Checkout page shows an empty cart even though the user added two "
        "items. The 'Place order' button is greyed out. A red banner reads "
        "'Payment method declined'."
    ),
    operation={
        "name": "op",
        "type": "choice",
        "options": with_abstain(["click", "fill", "wait", "escalate"]),
        "prompt": "What should the support agent do next about this checkout state?",
    },
    targets={
        "click": {
            "name": "click_target",
            "type": "choice",
            "options": ["retry-payment-button", "edit-cart-button", "contact-support-link"],
            "prompt": "Which control should be clicked?",
        },
        "fill": {
            "name": "fill_target",
            "type": "choice",
            "options": ["card-number-field", "coupon-field"],
            "prompt": "Which field should be filled?",
        },
    },
)

print(f"operation : {out['operation']} (conf {out['confidence']:.2f})")
print(f"target    : {out['target']} (conf {out['target_confidence']})")
print(f"latency   : {out['_meta']['latency_ms']} ms")
