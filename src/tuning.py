"""Nested time-series hyperparameter tuning and per-patient calibration.

Two things here:

1. :func:`tune_hyperparameters` — an **expanding-window** time-series CV over
   the development set (train+val, i.e. the *inner* loop of a nested scheme;
   the held-out test patients are the untouched *outer* loop). It grid-searches
   a small, sensible space and returns the params with the best mean CV RMSE.
   No future data or test patient ever leaks into selection.

2. :class:`PerPatientCalibrator` — adapts a population model to an individual
   by fitting a per-patient affine correction (slope/bias) on that patient's
   *earliest* data and applying it to their later data. This mirrors how you'd
   calibrate to a new user's first day without leaking their future.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import CALIB_FRACTION, HORIZONS_MIN, N_TSCV_FOLDS
from .models import make_boosted_regressor

# A small, laptop-friendly grid. Kept deliberately compact so nested CV runs
# in seconds; widen it for a real study.
DEFAULT_GRID: list[dict] = [
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05},
    {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05},
    {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.03},
    {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.05},
    {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.03},
]


def _expanding_folds(n: int, n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window splits over a time-ordered index of length ``n``."""
    folds = []
    step = n // (n_folds + 1)
    for k in range(1, n_folds + 1):
        train_end = step * k
        val_end = step * (k + 1) if k < n_folds else n
        if train_end < 1 or val_end <= train_end:
            continue
        folds.append((np.arange(0, train_end), np.arange(train_end, val_end)))
    return folds


def tune_hyperparameters(
    dev_df: pd.DataFrame,
    feature_cols: list[str],
    horizon_min: int,
    grid: list[dict] | None = None,
    n_folds: int = N_TSCV_FOLDS,
) -> tuple[dict, pd.DataFrame]:
    """Grid-search hyperparameters with expanding-window time-series CV.

    Selection is done on a single representative horizon (the caller passes
    the headline horizon) and the winning params are reused across horizons —
    a pragmatic choice that keeps tuning fast while staying leakage-free.

    Returns
    -------
    (best_params, results_table)
    """
    grid = grid or DEFAULT_GRID
    dev_sorted = dev_df.sort_values("timestamp").reset_index(drop=True)
    target = f"y_h{horizon_min}"
    folds = _expanding_folds(len(dev_sorted), n_folds)

    rows = []
    for params in grid:
        fold_rmse = []
        for tr_idx, va_idx in folds:
            tr, va = dev_sorted.iloc[tr_idx], dev_sorted.iloc[va_idx]
            model = make_boosted_regressor(params)
            model.fit(tr[feature_cols], tr[target])
            pred = model.predict(va[feature_cols])
            fold_rmse.append(float(np.sqrt(np.mean((va[target].to_numpy() - pred) ** 2))))
        rows.append({**params, "cv_rmse": float(np.mean(fold_rmse))})

    results = pd.DataFrame(rows).sort_values("cv_rmse").reset_index(drop=True)
    best = {k: results.iloc[0][k] for k in ("n_estimators", "max_depth", "learning_rate")}
    best = {"n_estimators": int(best["n_estimators"]), "max_depth": int(best["max_depth"]),
            "learning_rate": float(best["learning_rate"])}
    return best, results


@dataclass
class PerPatientCalibrator:
    """Per-patient affine recalibration of a base model's forecasts.

    For each patient, fit ``y ≈ a * yhat + b`` on their earliest
    ``calib_fraction`` of data (per horizon), then apply to the rest. Falls
    back to identity if a patient has too little calibration data.
    """

    calib_fraction: float = CALIB_FRACTION
    _coef: dict = field(default_factory=dict)  # (patient, horizon) -> (a, b)

    def fit_apply(
        self, base_model, test_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[int, np.ndarray], dict[int, np.ndarray]]:
        """Calibrate on each patient's early data; evaluate on the rest.

        Returns
        -------
        (eval_df, raw_preds_by_h, calibrated_preds_by_h)
            ``eval_df`` is the post-calibration slice actually scored; the two
            dicts are aligned to it (raw = population model, calibrated =
            per-patient-adjusted) so the caller can report the lift.
        """
        eval_parts = []
        raw_hold: dict[int, list] = {h: [] for h in HORIZONS_MIN}
        cal_hold: dict[int, list] = {h: [] for h in HORIZONS_MIN}

        for pid, sub in test_df.groupby("patient_id"):
            sub = sub.sort_values("timestamp").reset_index(drop=True)
            cut = int(len(sub) * self.calib_fraction)
            calib, evalp = sub.iloc[:cut], sub.iloc[cut:]
            if len(evalp) == 0:
                continue

            for h in HORIZONS_MIN:
                raw_eval = np.asarray(base_model.predict(evalp, h))
                a, b = self._gated_fit(base_model, calib, h)
                self._coef[(pid, h)] = (a, b)
                cal_hold[h].append(a * raw_eval + b)
                raw_hold[h].append(raw_eval)
            eval_parts.append(evalp)

        eval_df = pd.concat(eval_parts, ignore_index=True)
        raw = {h: np.concatenate(raw_hold[h]) for h in HORIZONS_MIN}
        cal = {h: np.concatenate(cal_hold[h]) for h in HORIZONS_MIN}
        return eval_df, raw, cal

    def _gated_fit(self, base_model, calib: pd.DataFrame, h: int) -> tuple[float, float]:
        """Fit affine calibration, but only keep it if it beats identity.

        The calibration window is itself split into fit / check halves. We
        accept the per-patient correction only when it lowers error on the
        held-out check half — a guard against negative transfer (with strongly
        autoregressive features, per-patient recalibration often adds little,
        so 'do no harm' matters). Otherwise we fall back to identity.
        """
        if len(calib) < 60:
            return 1.0, 0.0
        half = len(calib) // 2
        cfit, ccheck = calib.iloc[:half], calib.iloc[half:]
        a, b = _fit_affine(
            np.asarray(base_model.predict(cfit, h)), cfit[f"y_h{h}"].to_numpy()
        )
        # Bound the correction so a single non-stationary fit can't blow up
        # predictions (worst-case damage stays small even if the gate slips).
        a = float(np.clip(a, 0.9, 1.1))
        b = float(np.clip(b, -15.0, 15.0))
        chk_raw = np.asarray(base_model.predict(ccheck, h))
        y_chk = ccheck[f"y_h{h}"].to_numpy()
        err_id = np.mean((y_chk - chk_raw) ** 2)
        err_cal = np.mean((y_chk - (a * chk_raw + b)) ** 2)
        # Require a clear (not marginal) improvement on the held-out check half.
        return (a, b) if err_cal < 0.98 * err_id else (1.0, 0.0)


def _fit_affine(yhat: np.ndarray, y: np.ndarray, slope_shrink: float = 0.5) -> tuple[float, float]:
    """Bias-dominant affine calibration mapping ``yhat -> y``.

    The per-patient **bias** (mean offset) is the reliable signal from a short
    calibration window; the **slope** is noisier, so we shrink it halfway
    toward 1 and then set the intercept so the calibration-window means match.
    Returns ``(a, b)`` for ``y ~= a*yhat + b``.
    """
    yhat = np.asarray(yhat, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.var(yhat) < 1e-6:
        return 1.0, float(np.mean(y - yhat))
    a_raw, _ = np.polyfit(yhat, y, 1)
    if not (0.3 <= a_raw <= 3.0):
        a_raw = 1.0
    a = 1.0 + slope_shrink * (a_raw - 1.0)  # shrink slope toward identity
    b = float(np.mean(y) - a * np.mean(yhat))  # match the mean (bias correction)
    return float(a), b
