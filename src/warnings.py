"""Early-warning logic: turn forecasts into actionable alerts.

Rules
-----
* Hypo warning: a forecast at any horizon crosses **below** 70 mg/dL.
* Hyper warning: a forecast at any horizon crosses **above** 180 mg/dL.
* Hysteresis: require ``MIN_CONSECUTIVE_ALERTS`` consecutive threshold-
  crossing forecasts (same patient/segment/horizon/type) before firing, to
  suppress flickering alarms.

Each emitted warning carries: ``fired_at`` (when the alarm triggers),
``predicted_event_time`` (fired_at + horizon), ``kind`` (hypo/hyper),
``severity`` (level 1 / level 2) and the ``horizon_min`` that tripped it.
"""
from __future__ import annotations

import pandas as pd

from .config import (
    HORIZONS_MIN,
    HYPER_L1,
    HYPER_L2,
    HYPO_L1,
    HYPO_L2,
    MIN_CONSECUTIVE_ALERTS,
    SAMPLE_MINUTES,
)


def _severity(kind: str, value: float) -> str:
    if kind == "hypo":
        return "level2" if value < HYPO_L2 else "level1"
    return "level2" if value > HYPER_L2 else "level1"


def generate_warnings(
    df: pd.DataFrame,
    preds_by_horizon: dict[int, "pd.Series | list"],
    min_consecutive: int = MIN_CONSECUTIVE_ALERTS,
) -> pd.DataFrame:
    """Produce a table of warnings from per-horizon forecasts.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``patient_id``, ``segment``, ``timestamp`` aligned
        row-for-row with the prediction arrays.
    preds_by_horizon : dict[int, array-like]
        Horizon (minutes) -> predicted glucose for each row of ``df``.
    min_consecutive : int
        Hysteresis threshold (consecutive crossings required to fire).

    Returns
    -------
    pd.DataFrame with one row per fired warning.
    """
    work = df[["patient_id", "segment", "timestamp"]].reset_index(drop=True).copy()
    for h in HORIZONS_MIN:
        work[f"pred_h{h}"] = pd.Series(list(preds_by_horizon[h])).values

    records: list[dict] = []
    for (pid, seg), block in work.groupby(["patient_id", "segment"], sort=True):
        block = block.sort_values("timestamp").reset_index(drop=True)
        for h in HORIZONS_MIN:
            preds = block[f"pred_h{h}"].to_numpy()
            ts = block["timestamp"].to_numpy()
            for kind, crossed in (
                ("hypo", preds < HYPO_L1),
                ("hyper", preds > HYPER_L1),
            ):
                _emit_runs(
                    records, pid, seg, h, kind, crossed, preds, ts, min_consecutive
                )

    cols = [
        "patient_id",
        "segment",
        "kind",
        "severity",
        "horizon_min",
        "fired_at",
        "predicted_event_time",
        "predicted_value",
    ]
    if not records:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame.from_records(records)[cols].sort_values(
        ["patient_id", "fired_at"]
    ).reset_index(drop=True)


def _emit_runs(
    records: list[dict],
    pid,
    seg,
    horizon_min: int,
    kind: str,
    crossed,
    preds,
    ts,
    min_consecutive: int,
) -> None:
    """Fire once per run of >= min_consecutive consecutive crossings.

    We fire at the *first* sample that completes the hysteresis requirement,
    which is the earliest defensible alarm time for that episode.
    """
    run = 0
    fired_this_run = False
    for i, c in enumerate(crossed):
        if c:
            run += 1
            if run >= min_consecutive and not fired_this_run:
                fired_at = pd.Timestamp(ts[i])
                event_time = fired_at + pd.Timedelta(minutes=horizon_min)
                records.append(
                    {
                        "patient_id": pid,
                        "segment": seg,
                        "kind": kind,
                        "severity": _severity(kind, float(preds[i])),
                        "horizon_min": horizon_min,
                        "fired_at": fired_at,
                        "predicted_event_time": event_time,
                        "predicted_value": float(preds[i]),
                    }
                )
                fired_this_run = True
        else:
            run = 0
            fired_this_run = False


def generate_warnings_prob(
    df: pd.DataFrame,
    hypo_prob_by_horizon: dict[int, "pd.Series | list"],
    hyper_prob_by_horizon: dict[int, "pd.Series | list"],
    p_threshold: float,
    min_consecutive: int = MIN_CONSECUTIVE_ALERTS,
) -> pd.DataFrame:
    """Probabilistic warnings: fire when P(cross) exceeds ``p_threshold``.

    Same hysteresis structure as :func:`generate_warnings`, but the trigger is
    a calibrated crossing probability (from the conformal forecaster) rather
    than a point-forecast threshold cross. Each warning carries a
    ``confidence`` = the probability at the firing sample.
    """
    work = df[["patient_id", "segment", "timestamp"]].reset_index(drop=True).copy()
    for h in HORIZONS_MIN:
        work[f"hypo_p{h}"] = pd.Series(list(hypo_prob_by_horizon[h])).values
        work[f"hyper_p{h}"] = pd.Series(list(hyper_prob_by_horizon[h])).values

    records: list[dict] = []
    for (pid, seg), block in work.groupby(["patient_id", "segment"], sort=True):
        block = block.sort_values("timestamp").reset_index(drop=True)
        ts = block["timestamp"].to_numpy()
        for h in HORIZONS_MIN:
            for kind, col in (("hypo", f"hypo_p{h}"), ("hyper", f"hyper_p{h}")):
                probs = block[col].to_numpy()
                crossed = probs >= p_threshold
                _emit_prob_runs(
                    records, pid, seg, h, kind, crossed, probs, ts, min_consecutive
                )

    cols = [
        "patient_id",
        "segment",
        "kind",
        "severity",
        "horizon_min",
        "fired_at",
        "predicted_event_time",
        "confidence",
    ]
    if not records:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame.from_records(records)[cols].sort_values(
        ["patient_id", "fired_at"]
    ).reset_index(drop=True)


def _emit_prob_runs(
    records, pid, seg, horizon_min, kind, crossed, probs, ts, min_consecutive
) -> None:
    """Fire once per run of >= min_consecutive consecutive high-prob samples."""
    run = 0
    fired = False
    for i, c in enumerate(crossed):
        if c:
            run += 1
            if run >= min_consecutive and not fired:
                fired_at = pd.Timestamp(ts[i])
                records.append(
                    {
                        "patient_id": pid,
                        "segment": seg,
                        "kind": kind,
                        "severity": "high" if probs[i] >= 0.7 else "moderate",
                        "horizon_min": horizon_min,
                        "fired_at": fired_at,
                        "predicted_event_time": fired_at
                        + pd.Timedelta(minutes=horizon_min),
                        "confidence": float(probs[i]),
                    }
                )
                fired = True
        else:
            run = 0
            fired = False


def true_events(df: pd.DataFrame) -> pd.DataFrame:
    """Label ground-truth hypo/hyper episodes from actual glucose.

    An episode is a maximal run of samples on the wrong side of a threshold.
    Returns start/end times per episode, used to score the warnings.
    """
    records: list[dict] = []
    for (pid, seg), block in df.groupby(["patient_id", "segment"], sort=True):
        block = block.sort_values("timestamp").reset_index(drop=True)
        g = block["glucose"].to_numpy()
        ts = block["timestamp"].to_numpy()
        for kind, mask in (("hypo", g < HYPO_L1), ("hyper", g > HYPER_L1)):
            _collect_episodes(records, pid, seg, kind, mask, g, ts)
    cols = ["patient_id", "segment", "kind", "start", "end", "extreme_value"]
    if not records:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame.from_records(records)[cols]


def _collect_episodes(records, pid, seg, kind, mask, g, ts) -> None:
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            seg_vals = g[i : j + 1]
            extreme = seg_vals.min() if kind == "hypo" else seg_vals.max()
            records.append(
                {
                    "patient_id": pid,
                    "segment": seg,
                    "kind": kind,
                    "start": pd.Timestamp(ts[i]),
                    "end": pd.Timestamp(ts[j]),
                    "extreme_value": float(extreme),
                }
            )
            i = j + 1
        else:
            i += 1
