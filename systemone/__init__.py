"""systemone: local, open System One decision models.

A Jev-style typed-decision layer over local GLiClass checkpoints:

- api.SystemOne / api.systemone  -> choice / score / noul in one batched call
- calibration.*                 -> temperature / Platt / isotonic calibration
- mcp_server                     -> MCP tools (verify_claims, screen_content,
                                   rank_candidates) over stdio
- distill.*                      -> teacher -> tiny student pipeline
- tune.*                         -> domain fine-tuning helper

Quickstart:
    from systemone import SystemOne
    eng = SystemOne()  # loads gliclass-edge, one model at a time
    out = eng.systemone("The server is on fire", [
        {"name": "urgency", "type": "score", "levels": ["low", "medium", "high"]},
        {"name": "page", "type": "noul", "statement": "Should I page the on-call engineer?"},
    ])
"""

from .api import (
    ABSTAIN_LABEL,
    MAX_STATE_CHARS,
    LatencyStats,
    StallGuard,
    SystemOne,
    SystemOneError,
    default_device,
    make_questions,
    validate_choice,
    validate_distribution,
    with_abstain,
)
from .calibration import (
    CalibratedScorer,
    CalibrationExample,
    IsotonicCalibrator,
    PlattCalibrator,
    TemperatureCalibrator,
    expected_calibration_error,
    multiclass_ece,
)
from .shim import serve as serve_shim

__all__ = [
    "SystemOne",
    "SystemOneError",
    "MAX_STATE_CHARS",
    "ABSTAIN_LABEL",
    "LatencyStats",
    "StallGuard",
    "default_device",
    "make_questions",
    "validate_choice",
    "validate_distribution",
    "with_abstain",
    "serve_shim",
    "TemperatureCalibrator",
    "PlattCalibrator",
    "IsotonicCalibrator",
    "CalibratedScorer",
    "CalibrationExample",
    "expected_calibration_error",
    "multiclass_ece",
]

__version__ = "0.1.0"
