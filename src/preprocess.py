"""Preprocessing: uniform grid, gap handling, segmentation.

Steps, applied per patient:

1. Resample onto a uniform 5-min grid (mean within each bin).
2. Interpolate **short** gaps (<= MAX_INTERP_GAP_MIN) linearly.
3. Any remaining (longer) gap breaks the trace into ``segment`` blocks. We
   never interpolate, train, or predict across a segment boundary.

The result adds two columns:

    segment    : int   monotonically increasing id, unique per (patient, block)
    imputed    : bool  True where glucose was filled by interpolation
"""
from __future__ import annotations

import pandas as pd

from .config import MAX_INTERP_GAP_MIN, SAMPLE_MINUTES


def _resample_patient(df: pd.DataFrame) -> pd.DataFrame:
    """Put one patient onto a regular 5-min grid.

    Glucose is averaged within a bin; carbs and insulin are summed (dose
    totals); activity intensity is taken as the bin max.
    """
    freq = f"{SAMPLE_MINUTES}min"
    g = df.set_index("timestamp").sort_index()
    grid = pd.DataFrame(index=pd.date_range(g.index.min(), g.index.max(), freq=freq))
    grid.index.name = "timestamp"
    glucose = g["glucose"].resample(freq).mean()
    carbs = g["carbs"].resample(freq).sum().fillna(0.0)
    insulin = g["insulin"].resample(freq).sum().fillna(0.0)
    activity = g["activity"].resample(freq).max().fillna(0.0)
    out = grid.join(glucose).join(carbs).join(insulin).join(activity)
    out["patient_id"] = df["patient_id"].iloc[0]
    return out.reset_index()


def _fill_and_segment(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate short gaps; assign segment ids around long gaps."""
    max_gap_steps = MAX_INTERP_GAP_MIN // SAMPLE_MINUTES
    df = df.copy()
    df["imputed"] = df["glucose"].isna()

    # Linear interpolation with a hard limit so long gaps stay NaN.
    df["glucose"] = df["glucose"].interpolate(
        method="linear", limit=max_gap_steps, limit_direction="both"
    )
    # ``imputed`` should only mark cells we actually filled.
    df["imputed"] = df["imputed"] & df["glucose"].notna()

    # Long gaps remain NaN -> segment boundary. New segment starts after any
    # NaN run, and also at the very first valid row.
    is_valid = df["glucose"].notna()
    # A boundary is a valid row immediately preceded by an invalid row (or the
    # first row overall).
    boundary = is_valid & (~is_valid.shift(1, fill_value=False))
    df["segment"] = boundary.cumsum()
    df.loc[~is_valid, "segment"] = -1  # sentinel for dropped rows
    df = df[is_valid].copy()
    df["segment"] = df["segment"].astype(int)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full preprocessing pipeline over all patients.

    Returns a frame with unique, contiguous ``(patient_id, segment)`` blocks
    on a 5-min grid, ready for causal feature engineering.
    """
    parts = []
    seg_offset = 0
    for pid, sub in df.groupby("patient_id", sort=True):
        resampled = _resample_patient(sub)
        segmented = _fill_and_segment(resampled)
        if segmented.empty:
            continue
        # Make segment ids globally unique across patients.
        segmented["segment"] = segmented["segment"] + seg_offset
        seg_offset = int(segmented["segment"].max()) + 1
        parts.append(segmented)

    out = pd.concat(parts, ignore_index=True)
    return out[
        [
            "patient_id",
            "segment",
            "timestamp",
            "glucose",
            "carbs",
            "insulin",
            "activity",
            "imputed",
        ]
    ]
