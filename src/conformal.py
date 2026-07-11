"""Probabilistic forecasting: conformalized quantile regression (CQR).

Point forecasts don't tell a caregiver *how sure* the model is. This module
adds calibrated uncertainty:

* **Quantile regression** — a gradient-boosted model per quantile per horizon
  gives a predictive distribution for glucose at t+PH.
* **Conformalized quantile regression (CQR, Romano et al. 2019)** — a
  held-out calibration set adjusts the raw quantile band so the prediction
  interval has (approximately) the requested coverage, distribution-free.
* **Crossing probabilities** — from the predictive quantiles we read off
  ``P(glucose < 70)`` (hypo) and ``P(glucose > 180)`` (hyper) per sample, so
  each alarm can carry a confidence.

The quantile models use scikit-learn (pinball loss), so this works with no
GPU and no OpenMP.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    CONFORMAL_ALPHA,
    GLUCOSE_MAX,
    GLUCOSE_MIN,
    HORIZONS_MIN,
    QUANTILE_GRID,
)
from .models import make_boosted_regressor


class ConformalForecaster:
    """Per-horizon quantile models with CQR calibration and crossing probs."""

    def __init__(
        self,
        alpha: float = CONFORMAL_ALPHA,
        quantiles: tuple[float, ...] = QUANTILE_GRID,
        params: dict | None = None,
    ):
        self.alpha = alpha
        self.quantiles = tuple(sorted(quantiles))
        self.params = params
        self.feature_cols: list[str] = []
        self._models: dict[tuple[int, float], object] = {}
        self._cqr_adjust: dict[int, float] = {}
        # Lower/upper quantile levels bracketing the target interval.
        self._lo_q = alpha / 2
        self._hi_q = 1 - alpha / 2

    def fit(self, train_df: pd.DataFrame, feature_cols: list[str]) -> "ConformalForecaster":
        """Fit quantile regressors (incl. the interval edges) per horizon."""
        self.feature_cols = feature_cols
        levels = sorted(set(self.quantiles) | {self._lo_q, self._hi_q})
        X = train_df[feature_cols]
        for h in HORIZONS_MIN:
            y = train_df[f"y_h{h}"]
            for q in levels:
                model = make_boosted_regressor(self.params, quantile=q)
                model.fit(X, y)
                self._models[(h, round(q, 4))] = model
        return self

    def calibrate(self, calib_df: pd.DataFrame) -> "ConformalForecaster":
        """CQR: set the interval adjustment from calibration conformity scores.

        Score E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i)); the (1-alpha)
        empirical quantile of E becomes an additive width correction.
        """
        X = calib_df[self.feature_cols]
        for h in HORIZONS_MIN:
            y = calib_df[f"y_h{h}"].to_numpy()
            lo = self._models[(h, round(self._lo_q, 4))].predict(X)
            hi = self._models[(h, round(self._hi_q, 4))].predict(X)
            scores = np.maximum(lo - y, y - hi)
            n = len(scores)
            level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
            self._cqr_adjust[h] = float(np.quantile(scores, level, method="higher"))
        return self

    def predict_quantiles(self, df: pd.DataFrame, horizon_min: int) -> dict[float, np.ndarray]:
        """Predicted glucose at each fitted quantile (monotone-sorted)."""
        X = df[self.feature_cols]
        preds = {
            q: self._models[(horizon_min, round(q, 4))].predict(X) for q in self.quantiles
        }
        # Enforce monotonicity across quantiles (guards crossing quantiles).
        grid = np.array(self.quantiles)
        stacked = np.vstack([preds[q] for q in self.quantiles])
        stacked = np.maximum.accumulate(stacked, axis=0)
        return {q: stacked[i] for i, q in enumerate(grid)}

    def predict_interval(
        self, df: pd.DataFrame, horizon_min: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Conformalized (lower, upper) prediction interval for the horizon."""
        X = df[self.feature_cols]
        adj = self._cqr_adjust.get(horizon_min, 0.0)
        lo = self._models[(horizon_min, round(self._lo_q, 4))].predict(X) - adj
        hi = self._models[(horizon_min, round(self._hi_q, 4))].predict(X) + adj
        return np.clip(lo, GLUCOSE_MIN, GLUCOSE_MAX), np.clip(hi, GLUCOSE_MIN, GLUCOSE_MAX)

    def predict_median(self, df: pd.DataFrame, horizon_min: int) -> np.ndarray:
        """Median (0.5-quantile) point forecast, if available."""
        key = (horizon_min, 0.5)
        if key in self._models:
            return self._models[key].predict(df[self.feature_cols])
        q = self.predict_quantiles(df, horizon_min)
        mid = self.quantiles[len(self.quantiles) // 2]
        return q[mid]

    def crossing_prob(
        self, df: pd.DataFrame, horizon_min: int, threshold: float, direction: str
    ) -> np.ndarray:
        """P(glucose crosses ``threshold``) at the horizon, per sample.

        Builds an empirical CDF from the predicted quantiles (anchored at the
        physiological glucose bounds for 0 and 1) and reads off the tail
        probability. ``direction`` is ``"below"`` (hypo) or ``"above"`` (hyper).
        """
        q = self.predict_quantiles(df, horizon_min)
        grid = np.array(self.quantiles)
        vals = np.vstack([q[qi] for qi in self.quantiles]).T  # (N, Q)
        n = vals.shape[0]

        # Anchor CDF at physiological bounds so probabilities go to 0/1 in tails.
        lo_anchor = np.full((n, 1), GLUCOSE_MIN)
        hi_anchor = np.full((n, 1), GLUCOSE_MAX)
        x = np.hstack([lo_anchor, vals, hi_anchor])
        p = np.concatenate([[0.0], grid, [1.0]])

        out = np.empty(n)
        for i in range(n):
            xi = np.maximum.accumulate(x[i])
            cdf_at = float(np.interp(threshold, xi, p))
            out[i] = cdf_at if direction == "below" else (1.0 - cdf_at)
        return np.clip(out, 0.0, 1.0)

    def hypo_hyper_probs(
        self, df: pd.DataFrame, hypo_thr: float, hyper_thr: float
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        """Convenience: P(hypo) and P(hyper) per horizon over a frame."""
        hypo = {
            h: self.crossing_prob(df, h, hypo_thr, "below") for h in HORIZONS_MIN
        }
        hyper = {
            h: self.crossing_prob(df, h, hyper_thr, "above") for h in HORIZONS_MIN
        }
        return hypo, hyper


def interval_metrics(
    lower: np.ndarray, upper: np.ndarray, y_true: np.ndarray
) -> dict[str, float]:
    """Empirical coverage and mean width of a prediction interval."""
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    y_true = np.asarray(y_true)
    covered = (y_true >= lower) & (y_true <= upper)
    return {
        "coverage": float(np.mean(covered)),
        "mean_width": float(np.mean(upper - lower)),
    }
