"""Causal feature engineering.

All features are computed from **past and present** glucose only — nothing
here peeks at the future. Features are built strictly within a
``(patient_id, segment)`` block so we never reach across a gap. Targets
(glucose at t+PH) are likewise only defined when the future sample exists in
the same segment.

Feature families:
  * lagged glucose over the lookback window
  * first/second differences (rate of change, acceleration)
  * rolling mean/std over short and medium windows
  * time-of-day sin/cos (circadian)
  * optional exogenous meal/insulin recency flags
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    ACT_TAU_MIN,
    COB_TAU_MIN,
    HORIZONS_MIN,
    IOB_TAU_MIN,
    LOOKBACK_MIN,
    SAMPLE_MINUTES,
    horizon_steps,
)

# Lags in steps across the lookback window (5..LOOKBACK minutes back).
_LAG_STEPS = list(range(1, LOOKBACK_MIN // SAMPLE_MINUTES + 1))
_ROLL_WINDOWS = [3, 6, 12]  # 15, 30, 60 min rolling stats


def _target_columns() -> list[str]:
    return [f"y_h{h}" for h in HORIZONS_MIN]


def build_features(df: pd.DataFrame, use_exogenous: bool = True) -> pd.DataFrame:
    """Turn a preprocessed frame into a model-ready feature matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`preprocess.preprocess`.
    use_exogenous : bool
        Include meal/insulin recency features if the flags carry signal.

    Returns
    -------
    pd.DataFrame with the original id columns, engineered features, and one
    target column per horizon (``y_h15`` etc.). Rows lacking full lookback or
    a valid future target are dropped.
    """
    frames = []
    for (_pid, _seg), block in df.groupby(["patient_id", "segment"], sort=True):
        block = block.sort_values("timestamp").reset_index(drop=True)
        if len(block) <= max(_LAG_STEPS) + max(horizon_steps(h) for h in HORIZONS_MIN):
            continue
        frames.append(_features_for_block(block, use_exogenous))
    if not frames:
        raise ValueError("No segment was long enough to build features.")
    out = pd.concat(frames, ignore_index=True)
    return out


def _features_for_block(block: pd.DataFrame, use_exogenous: bool) -> pd.DataFrame:
    g = block["glucose"].astype(float)
    feat = pd.DataFrame(index=block.index)
    feat["patient_id"] = block["patient_id"].values
    feat["segment"] = block["segment"].values
    feat["timestamp"] = block["timestamp"].values
    feat["glucose"] = g.values  # current value == persistence baseline input

    # --- Lagged values ----------------------------------------------------
    for k in _LAG_STEPS:
        feat[f"lag_{k}"] = g.shift(k)

    # --- Rate of change & acceleration ------------------------------------
    d1 = g.diff()
    feat["roc"] = d1                     # per 5 min
    feat["roc_2"] = g.diff(2)            # per 10 min
    feat["accel"] = d1.diff()           # second difference

    # --- Rolling statistics (causal: use only past+present) ---------------
    for w in _ROLL_WINDOWS:
        feat[f"roll_mean_{w}"] = g.rolling(w, min_periods=w).mean()
        feat[f"roll_std_{w}"] = g.rolling(w, min_periods=w).std()
    feat["roll_min_12"] = g.rolling(12, min_periods=12).min()
    feat["roll_max_12"] = g.rolling(12, min_periods=12).max()

    # --- Time-of-day (circadian) ------------------------------------------
    minute_of_day = (
        block["timestamp"].dt.hour * 60 + block["timestamp"].dt.minute
    ).values
    ang = 2 * np.pi * minute_of_day / (24 * 60)
    feat["tod_sin"] = np.sin(ang)
    feat["tod_cos"] = np.cos(ang)

    # --- Exogenous inputs: insulin/carbs-on-board & activity --------------
    # Always emitted (constant 0 when a channel is absent) so the feature set
    # is stable across data sources. All are causal decaying accumulations.
    if use_exogenous:
        carbs = block.get("carbs", pd.Series(0.0, index=block.index)).to_numpy()
        insulin = block.get("insulin", pd.Series(0.0, index=block.index)).to_numpy()
        activity = block.get("activity", pd.Series(0.0, index=block.index)).to_numpy()

        feat["iob"] = _decay_accumulate(insulin, IOB_TAU_MIN)
        feat["cob"] = _decay_accumulate(carbs, COB_TAU_MIN)
        feat["activity_load"] = _decay_accumulate(activity, ACT_TAU_MIN)
        feat["insulin_30"] = _rolling_sum(insulin, 6)   # units in last 30 min
        feat["carbs_30"] = _rolling_sum(carbs, 6)       # grams in last 30 min
        feat["activity_recent"] = pd.Series(activity).rolling(6, min_periods=1).max().values
        feat["mins_since_carbs"] = _minutes_since_event(carbs > 0)
        feat["mins_since_insulin"] = _minutes_since_event(insulin > 0)

    # --- Targets: glucose at t + PH, same segment only --------------------
    for h in HORIZONS_MIN:
        feat[f"y_h{h}"] = g.shift(-horizon_steps(h))

    # Drop warm-up rows (incomplete lookback) and rows with no future target.
    feat = feat.dropna().reset_index(drop=True)
    return feat


def _minutes_since_event(flags: np.ndarray) -> np.ndarray:
    """Minutes elapsed since the most recent event flag (capped at 24h)."""
    flags = np.asarray(flags) > 0
    out = np.full(len(flags), 24 * 60, dtype=float)
    last = -1
    for i, f in enumerate(flags):
        if f:
            last = i
        if last >= 0:
            out[i] = (i - last) * SAMPLE_MINUTES
    return np.clip(out, 0, 24 * 60)


def _decay_accumulate(x: np.ndarray, tau_min: float) -> np.ndarray:
    """Causal exponentially-decaying accumulation (an 'on-board' quantity).

    ``a[t] = a[t-1] * exp(-dt/tau) + x[t]`` — e.g. insulin-on-board from an
    insulin delivery series. Uses only past and present values.
    """
    x = np.asarray(x, dtype=float)
    decay = float(np.exp(-SAMPLE_MINUTES / tau_min))
    out = np.zeros(len(x))
    acc = 0.0
    for i, v in enumerate(x):
        acc = acc * decay + v
        out[i] = acc
    return out


def _rolling_sum(x: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling sum over ``window`` steps (partial windows allowed)."""
    return pd.Series(np.asarray(x, dtype=float)).rolling(window, min_periods=1).sum().values


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model input columns = everything except ids, timestamp and targets."""
    exclude = {"patient_id", "segment", "timestamp"} | set(_target_columns())
    return [c for c in df.columns if c not in exclude]
