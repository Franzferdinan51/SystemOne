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
    build_decision_prompts,
    choice_confidence,
    default_device,
    make_questions,
    noul_confidence,
    score_confidence,
    validate_choice,
    validate_distribution,
    with_abstain,
)
from .calibration import (
    DECISION_TYPES,
    MIN_ROWS_PER_TYPE,
    CalibratedScorer,
    CalibrationExample,
    IsotonicCalibrator,
    PerTypeTemperatureCalibrator,
    PlattCalibrator,
    TemperatureCalibrator,
    expected_calibration_error,
    fit_temperature_by_type,
    load_type_calibration,
    multiclass_ece,
)
from .metrics import (
    aurc,
    brier_score,
    format_table,
    nll_score,
    selective_accuracy,
    summarize,
    top_label_ece,
)
from .shim import serve as serve_shim
from .sglang_backend import SGLangBackend, SGLangError

__all__ = [
    "SystemOne",
    "SystemOneError",
    "SGLangBackend",
    "SGLangError",
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
    "PerTypeTemperatureCalibrator",
    "fit_temperature_by_type",
    "load_type_calibration",
    "DECISION_TYPES",
    "MIN_ROWS_PER_TYPE",
    "choice_confidence",
    "score_confidence",
    "noul_confidence",
    "build_decision_prompts",
    "top_label_ece",
    "brier_score",
    "nll_score",
    "aurc",
    "selective_accuracy",
    "summarize",
    "format_table",
]

__version__ = "0.1.0"
