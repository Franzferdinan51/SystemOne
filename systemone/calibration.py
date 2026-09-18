"""Probability calibration for GLiClass raw scores.

GLiClass returns raw per-label scores that are not honest probabilities
(typically overconfident). This module implements standard post-hoc
calibration methods — the practical stand-in for Jev's RLCD-trained
calibrated confidence:

- TemperatureCalibrator: single temperature T fit by NLL minimization,
  applied as softmax(scores / T). Best for multi-class (choice/score).
- PlattCalibrator: 1-D logistic regression on the positive-class score.
  Best for binary decisions (noul, safe/unsafe).
- IsotonicCalibrator: non-parametric isotonic regression. More flexible
  than Platt when you have enough calibration data (a few hundred points).

Also provides expected_calibration_error() to measure miscalibration
before/after.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

import numpy as np

try:
    from scipy.optimize import minimize_scalar
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def expected_calibration_error(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error for binary/confidence predictions.

    Bins predictions by confidence; ECE = sum_b |acc_b - conf_b| * (n_b / n).
    A perfectly calibrated model has ECE ~ 0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    assert y_true.shape == y_prob.shape, "y_true and y_prob must match"
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # last bin is inclusive on the right so p=1.0 lands somewhere
        mask = (y_prob > lo) & (y_prob <= hi) if i else (y_prob >= lo) & (y_prob <= hi)
        n_b = mask.sum()
        if n_b == 0:
            continue
        acc_b = y_true[mask].mean()
        conf_b = y_prob[mask].mean()
        ece += (n_b / len(y_true)) * abs(acc_b - conf_b)
    return float(ece)


def multiclass_ece(
    y_true: Sequence[int],
    proba: Sequence[Sequence[float]],
    n_bins: int = 10,
) -> float:
    """ECE over the predicted (max-probability) class for multi-class outputs."""
    proba = np.asarray(proba, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    return expected_calibration_error(correct, conf, n_bins=n_bins)


class TemperatureCalibrator:
    """Single-parameter temperature scaling: p = softmax(scores / T).

    Fit T by minimizing negative log-likelihood on a labeled calibration
    set. T > 1 softens overconfident scores; T < 1 sharpens underconfident
    ones. Requires scipy.
    """

    def __init__(self) -> None:
        self.temperature_: float = 1.0
        self.fitted_: bool = False

    def fit(self, scores: Sequence[Sequence[float]], labels: Sequence[int]) -> "TemperatureCalibrator":
        if not _HAS_SCIPY:
            raise ImportError("scipy is required for TemperatureCalibrator.fit()")
        S = np.asarray(scores, dtype=np.float64)
        y = np.asarray(labels, dtype=int)
        n = len(y)

        def nll(logT: float) -> float:
            T = float(np.exp(logT))
            P = softmax(S / T)
            # gather predicted probability of the true class
            p_true = P[np.arange(n), y]
            return float(-np.log(np.clip(p_true, 1e-12, 1.0)).mean())

        res = minimize_scalar(nll, bounds=(-4.0, 4.0), method="bounded",
                              options={"xatol": 1e-4})
        self.temperature_ = float(np.exp(res.x))
        self.fitted_ = True
        return self

    def predict_proba(self, scores: Sequence[Sequence[float]]) -> np.ndarray:
        return softmax(np.asarray(scores, dtype=np.float64) / self.temperature_)


class PlattCalibrator:
    """Binary Platt scaling: p = sigmoid(a * score + b).

    Fit with logistic regression on the positive-class score.
    Requires scikit-learn.
    """

    def __init__(self) -> None:
        self.model_: LogisticRegression | None = None

    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> "PlattCalibrator":
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required for PlattCalibrator.fit()")
        X = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        y = np.asarray(labels, dtype=int)
        self.model_ = LogisticRegression(max_iter=1000)
        self.model_.fit(X, y)
        return self

    def predict_proba(self, scores: Sequence[float]) -> np.ndarray:
        assert self.model_ is not None, "call fit() first"
        X = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        return self.model_.predict_proba(X)[:, 1]


class IsotonicCalibrator:
    """Binary isotonic regression calibration (non-parametric).

    More flexible than Platt scaling but needs more calibration data
    (a few hundred points) to avoid overfitting. Requires scikit-learn.
    """

    def __init__(self) -> None:
        self.model_: IsotonicRegression | None = None

    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> "IsotonicCalibrator":
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required for IsotonicCalibrator.fit()")
        X = np.asarray(scores, dtype=np.float64)
        y = np.asarray(labels, dtype=int)
        self.model_ = IsotonicRegression(out_of_bounds="clip")
        self.model_.fit(X, y)
        return self

    def predict_proba(self, scores: Sequence[float]) -> np.ndarray:
        assert self.model_ is not None, "call fit() first"
        X = np.asarray(scores, dtype=np.float64)
        return np.clip(self.model_.predict(X), 0.0, 1.0)


@dataclass
class CalibrationExample:
    text: str
    labels: List[str]
    true_label: str


class CalibratedScorer:
    """Wraps a raw score_fn with a fitted calibrator.

    score_fn: callable (texts: List[str], labels: List[str]) -> List[List[float]]
              returning raw per-label scores for each text.
    """

    def __init__(
        self,
        score_fn: Callable[[List[str], List[str]], List[List[float]]],
        method: str = "temperature",
    ) -> None:
        if method == "temperature":
            self.calibrator: TemperatureCalibrator | PlattCalibrator | IsotonicCalibrator = TemperatureCalibrator()
        elif method == "platt":
            self.calibrator = PlattCalibrator()
        elif method == "isotonic":
            self.calibrator = IsotonicCalibrator()
        else:
            raise ValueError(f"unknown calibration method: {method!r}")
        self.score_fn = score_fn
        self.method = method

    def fit(self, examples: Sequence[CalibrationExample]) -> "CalibratedScorer":
        texts = [e.text for e in examples]
        label_lists = [e.labels for e in examples]
        # score each example against its own label set
        raw: List[List[float]] = []
        for t, labs in zip(texts, label_lists):
            raw.append(self.score_fn([t], labs)[0])
        if self.method == "temperature":
            y_idx = [labs.index(e.true_label) for e, labs in zip(examples, label_lists)]
            self.calibrator.fit(raw, y_idx)
        else:
            # binary: probability of the true label being rank-1 vs not.
            # We calibrate P(correct top-1) using the top-1 raw score.
            assert all(len(e.labels) == 2 for e in examples), \
                "platt/isotonic need binary (2-label) examples; use temperature for multi-class"
            pos_scores = [max(r) for r in raw]
            # label = 1 if the top-scoring label is the true label
            y_bin = [int(labs[int(np.argmax(r))] == e.true_label)
                     for e, labs, r in zip(examples, label_lists, raw)]
            self.calibrator.fit(pos_scores, y_bin)
        return self

    def predict_proba(self, texts: List[str], labels: List[str]) -> np.ndarray:
        raw = self.score_fn(texts, labels)
        if self.method == "temperature":
            return self.calibrator.predict_proba(raw)
        # binary calibrators return P(top-1 is correct); distribute the
        # remainder uniformly over the other labels to keep a valid simplex.
        raw = np.asarray(raw, dtype=float)
        top1 = raw.argmax(axis=1)
        p_top1 = self.calibrator.predict_proba(raw.max(axis=1))
        k = raw.shape[1]
        out = np.full_like(raw, 0.0)
        for i in range(len(raw)):
            out[i, top1[i]] = p_top1[i]
            rest = [j for j in range(k) if j != top1[i]]
            if rest:
                out[i, rest] = (1.0 - p_top1[i]) / len(rest)
        return out
