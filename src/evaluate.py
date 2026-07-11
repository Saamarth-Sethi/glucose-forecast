"""Evaluation: regression metrics, Clarke Error Grid, event detection, plots.

This module is where the resume line gets made real. It reports, per horizon
and per model:

* RMSE and MAE (mg/dL)
* Clarke Error Grid Analysis — % of points in zones A-E, plus a scatter plot
* Event-detection metrics for the early-warning system — sensitivity,
  specificity, precision, F1 (sample-level, per horizon), and median lead
  time + false alarms per day (from the hysteresis warning system)

Plus three saved PNGs: a predicted-vs-actual overlay with warning markers, the
Clarke grid, and an episode/warning timeline.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless: save PNGs without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import (
    HORIZONS_MIN,
    HYPER_L1,
    HYPO_L1,
    OUTPUT_DIR,
)
from .warnings import generate_warnings, true_events

# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


# ---------------------------------------------------------------------------
# Clarke Error Grid Analysis
# ---------------------------------------------------------------------------


def clarke_zone(ref: float, pred: float) -> str:
    """Return the Clarke Error Grid zone ('A'-'E') for one point (mg/dL).

    Implements the standard Clarke et al. (1987) zone boundaries widely used
    for CGM accuracy assessment. ``ref`` is the true (reference) glucose,
    ``pred`` the predicted value.
    """
    # Zone A: clinically accurate (within 20%, or both in hypo range).
    if (ref <= 70 and pred <= 70) or (0.8 * ref <= pred <= 1.2 * ref):
        return "A"
    # Zone E: erroneous — opposite treatment.
    if (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180):
        return "E"
    # Zone C: overcorrection.
    if ((70 <= ref <= 290) and pred >= ref + 110) or (
        (130 <= ref <= 180) and pred <= (7.0 / 5.0) * ref - 182
    ):
        return "C"
    # Zone D: dangerous failure to detect.
    if (ref >= 240 and 70 <= pred <= 180) or (ref <= 70 and 70 <= pred <= 180):
        return "D"
    # Zone B: benign error.
    return "B"


def clarke_zone_percentages(
    ref: np.ndarray, pred: np.ndarray
) -> dict[str, float]:
    """Percent of points falling in each Clarke zone A-E."""
    counts = {z: 0 for z in "ABCDE"}
    for r, p in zip(np.asarray(ref), np.asarray(pred)):
        counts[clarke_zone(float(r), float(p))] += 1
    n = max(1, len(ref))
    return {z: 100.0 * counts[z] / n for z in "ABCDE"}


# ---------------------------------------------------------------------------
# Event-detection metrics (sample level, per horizon)
# ---------------------------------------------------------------------------


def _binary_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = (
        2 * prec * sens / (prec + sens)
        if prec == prec and sens == sens and (prec + sens) > 0
        else float("nan")
    )
    return {
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def event_detection_metrics(
    actual_future: np.ndarray, pred_future: np.ndarray, kind: str
) -> dict[str, float]:
    """Sample-level detection of an upcoming threshold crossing.

    ``actual_future``/``pred_future`` are the true and predicted glucose at
    t+PH. A positive means the forecast horizon lands in the event zone.
    """
    if kind == "hypo":
        yt = np.asarray(actual_future) < HYPO_L1
        yp = np.asarray(pred_future) < HYPO_L1
    else:
        yt = np.asarray(actual_future) > HYPER_L1
        yp = np.asarray(pred_future) > HYPER_L1
    return _binary_scores(yt, yp)


# ---------------------------------------------------------------------------
# Warning-system metrics (episode level: lead time & false alarms/day)
# ---------------------------------------------------------------------------


def observation_days(df: pd.DataFrame) -> float:
    """Total observation duration across patients, in days."""
    total = pd.Timedelta(0)
    for _pid, sub in df.groupby("patient_id"):
        total += sub["timestamp"].max() - sub["timestamp"].min()
    return max(total.total_seconds() / 86400.0, 1e-9)


def score_warnings(
    warns: pd.DataFrame,
    events: pd.DataFrame,
    days: float,
    match_tolerance_min: int = 30,
) -> dict[str, float]:
    """Score an arbitrary warnings table against ground-truth episodes.

    Detection: a true episode counts as caught if a same-kind warning fired at
    or before its onset (and no earlier than max-horizon + tolerance before).
    Lead time = onset - earliest matching alarm. False alarms are fired
    warnings whose predicted event never materialises within tolerance.
    Works for both deterministic and probabilistic warning tables.
    """
    tol = pd.Timedelta(minutes=match_tolerance_min)
    lead_window = pd.Timedelta(minutes=max(HORIZONS_MIN)) + tol
    out: dict[str, float] = {}

    for kind in ("hypo", "hyper"):
        ev = events[events["kind"] == kind]
        wn = warns[warns["kind"] == kind]

        detected = 0
        lead_times: list[float] = []
        for _, e in ev.iterrows():
            cand = wn[
                (wn["patient_id"] == e["patient_id"])
                & (wn["segment"] == e["segment"])
                & (wn["fired_at"] <= e["start"])
                & (wn["fired_at"] >= e["start"] - lead_window)
            ]
            if len(cand) > 0:
                detected += 1
                lead_times.append((e["start"] - cand["fired_at"].min()).total_seconds() / 60.0)

        false_alarms = 0
        for _, w in wn.iterrows():
            hit = ev[
                (ev["patient_id"] == w["patient_id"])
                & (ev["segment"] == w["segment"])
                & (ev["start"] >= w["fired_at"] - tol)
                & (ev["start"] <= w["predicted_event_time"] + tol)
            ]
            if len(hit) == 0:
                false_alarms += 1

        n_ev = len(ev)
        out[f"{kind}_episodes"] = n_ev
        out[f"{kind}_detected"] = detected
        out[f"{kind}_episode_sensitivity"] = detected / n_ev if n_ev else float("nan")
        out[f"{kind}_median_lead_min"] = (
            float(np.median(lead_times)) if lead_times else float("nan")
        )
        out[f"{kind}_false_alarms_per_day"] = false_alarms / days
    return out


def warning_system_metrics(
    test_df: pd.DataFrame,
    preds_by_horizon: dict[int, np.ndarray],
    match_tolerance_min: int = 30,
) -> dict[str, float]:
    """Operational metrics from the deterministic hysteresis warning system."""
    warns = generate_warnings(test_df, preds_by_horizon)
    events = true_events(test_df)
    return score_warnings(warns, events, observation_days(test_df), match_tolerance_min)


# ---------------------------------------------------------------------------
# Prediction assembly
# ---------------------------------------------------------------------------


def predict_all_horizons(model, df: pd.DataFrame) -> dict[int, np.ndarray]:
    """Return {horizon: predictions} for a model over a feature frame."""
    return {h: np.asarray(model.predict(df, h)) for h in HORIZONS_MIN}


def evaluate_model(model, test_df: pd.DataFrame) -> pd.DataFrame:
    """Per-horizon regression + event metrics for one model on the test set."""
    rows = []
    preds = predict_all_horizons(model, test_df)
    for h in HORIZONS_MIN:
        y_true = test_df[f"y_h{h}"].to_numpy()
        y_pred = preds[h]
        hypo = event_detection_metrics(y_true, y_pred, "hypo")
        hyper = event_detection_metrics(y_true, y_pred, "hyper")
        zones = clarke_zone_percentages(y_true, y_pred)
        rows.append(
            {
                "model": model.name,
                "horizon_min": h,
                "RMSE": rmse(y_true, y_pred),
                "MAE": mae(y_true, y_pred),
                "ClarkeA%": zones["A"],
                "ClarkeB%": zones["B"],
                "ClarkeA+B%": zones["A"] + zones["B"],
                "hypo_sens": hypo["sensitivity"],
                "hypo_spec": hypo["specificity"],
                "hypo_prec": hypo["precision"],
                "hypo_f1": hypo["f1"],
                "hyper_sens": hyper["sensitivity"],
                "hyper_spec": hyper["specificity"],
                "hyper_prec": hyper["precision"],
                "hyper_f1": hyper["f1"],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _ensure_outdir() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def plot_prediction_overlay(
    model,
    test_df: pd.DataFrame,
    horizon_min: int = 30,
    filename: str = "prediction_overlay.png",
) -> str:
    """Predicted-vs-actual overlay for one sample day, with warning markers."""
    outdir = _ensure_outdir()
    # Pick the (patient, segment) with the most rows for a clean, long trace.
    sizes = test_df.groupby(["patient_id", "segment"]).size()
    pid, seg = sizes.idxmax()
    block = test_df[(test_df.patient_id == pid) & (test_df.segment == seg)].copy()
    block = block.sort_values("timestamp").reset_index(drop=True)
    # Cap to ~one day for readability.
    block = block.iloc[: min(len(block), 24 * 12)]

    preds = model.predict(block, horizon_min)
    step = horizon_min // 5
    # Forecast made at t is a prediction FOR time t+PH: shift for alignment.
    pred_time = block["timestamp"] + pd.Timedelta(minutes=horizon_min)

    warns = generate_warnings(block, predict_all_horizons(model, block))

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(block["timestamp"], block["glucose"], label="Actual glucose", color="#1f77b4")
    ax.plot(
        pred_time,
        preds,
        label=f"Forecast (+{horizon_min} min)",
        color="#ff7f0e",
        alpha=0.85,
        linewidth=1.4,
    )
    ax.axhline(HYPO_L1, color="#2ca02c", ls="--", lw=1, label="Hypo 70")
    ax.axhline(HYPER_L1, color="#d62728", ls="--", lw=1, label="Hyper 180")

    for _, w in warns.iterrows():
        color = "#2ca02c" if w["kind"] == "hypo" else "#d62728"
        ax.axvline(w["fired_at"], color=color, alpha=0.35, lw=1)
    if len(warns):
        ax.scatter(
            warns["fired_at"],
            [ax.get_ylim()[1] * 0.98] * len(warns),
            marker="v",
            color=[("#2ca02c" if k == "hypo" else "#d62728") for k in warns["kind"]],
            s=40,
            label="Warning fired",
            zorder=5,
        )

    ax.set_title(f"Glucose forecast vs actual — patient {pid}, +{horizon_min} min horizon")
    ax.set_xlabel("Time")
    ax.set_ylabel("Glucose (mg/dL)")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    fig.tight_layout()
    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_forecast_intervals(
    conformal,
    test_df: pd.DataFrame,
    horizon_min: int = 30,
    filename: str = "forecast_intervals.png",
) -> str:
    """Actual vs. median forecast with the conformal prediction band."""
    outdir = _ensure_outdir()
    sizes = test_df.groupby(["patient_id", "segment"]).size()
    pid, seg = sizes.idxmax()
    block = test_df[(test_df.patient_id == pid) & (test_df.segment == seg)].copy()
    block = block.sort_values("timestamp").reset_index(drop=True)
    block = block.iloc[: min(len(block), 24 * 12)]

    median = conformal.predict_median(block, horizon_min)
    lower, upper = conformal.predict_interval(block, horizon_min)
    pred_time = block["timestamp"] + pd.Timedelta(minutes=horizon_min)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(block["timestamp"], block["glucose"], color="#1f77b4", label="Actual glucose")
    ax.plot(pred_time, median, color="#ff7f0e", lw=1.4, label=f"Median forecast (+{horizon_min}m)")
    ax.fill_between(
        pred_time, lower, upper, color="#ff7f0e", alpha=0.2,
        label="Conformal 80% interval",
    )
    ax.axhline(HYPO_L1, color="#2ca02c", ls="--", lw=1, label="Hypo 70")
    ax.axhline(HYPER_L1, color="#d62728", ls="--", lw=1, label="Hyper 180")
    ax.set_title(
        f"Probabilistic forecast with calibrated interval — patient {pid}, +{horizon_min} min"
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Glucose (mg/dL)")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    fig.tight_layout()
    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_clarke_grid(
    ref: np.ndarray,
    pred: np.ndarray,
    title: str = "Clarke Error Grid",
    filename: str = "clarke_grid.png",
) -> str:
    """Scatter of predicted vs reference with Clarke zone boundaries."""
    outdir = _ensure_outdir()
    ref = np.asarray(ref)
    pred = np.asarray(pred)
    pct = clarke_zone_percentages(ref, pred)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(ref, pred, s=4, alpha=0.25, color="#1f77b4")
    ax.plot([0, 400], [0, 400], "k:", lw=1)

    # Standard Clarke boundary lines.
    ax.plot([0, 175 / 3], [70, 70], "k-", lw=1)
    ax.plot([175 / 3, 320], [70, 400], "k-", lw=1)
    ax.plot([70, 70], [84, 400], "k-", lw=1)
    ax.plot([0, 70], [180, 180], "k-", lw=1)
    ax.plot([70, 290], [180, 400], "k-", lw=1)
    ax.plot([70, 70], [0, 56], "k-", lw=1)
    ax.plot([70, 400], [56, 320], "k-", lw=1)
    ax.plot([180, 180], [0, 70], "k-", lw=1)
    ax.plot([180, 400], [70, 70], "k-", lw=1)
    ax.plot([240, 240], [70, 180], "k-", lw=1)
    ax.plot([240, 400], [180, 180], "k-", lw=1)
    ax.plot([130, 180], [0, 70], "k-", lw=1)

    # Zone letters at canonical positions.
    for x, y, z in [
        (30, 15, "A"),
        (370, 260, "B"),
        (280, 370, "B"),
        (160, 370, "C"),
        (160, 15, "C"),
        (30, 140, "D"),
        (370, 120, "D"),
        (30, 370, "E"),
        (370, 15, "E"),
    ]:
        ax.text(x, y, z, fontsize=13, fontweight="bold", color="gray")

    caption = "  ".join(f"{z}:{pct[z]:.1f}%" for z in "ABCDE")
    ax.set_title(f"{title}\n{caption}", fontsize=11)
    ax.set_xlabel("Reference glucose (mg/dL)")
    ax.set_ylabel("Predicted glucose (mg/dL)")
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_aspect("equal")
    fig.tight_layout()
    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_warning_timeline(
    model,
    test_df: pd.DataFrame,
    filename: str = "warning_timeline.png",
) -> str:
    """Timeline of true episodes vs where warnings fired, for one patient."""
    outdir = _ensure_outdir()
    sizes = test_df.groupby("patient_id").size()
    pid = sizes.idxmax()
    block = test_df[test_df.patient_id == pid].copy().sort_values("timestamp")

    preds = predict_all_horizons(model, block)
    warns = generate_warnings(block, preds)
    events = true_events(block)

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(block["timestamp"], block["glucose"], color="#444", lw=0.8, label="Glucose")
    ax.axhline(HYPO_L1, color="#2ca02c", ls="--", lw=1)
    ax.axhline(HYPER_L1, color="#d62728", ls="--", lw=1)

    for _, e in events.iterrows():
        color = "#2ca02c" if e["kind"] == "hypo" else "#d62728"
        ax.axvspan(e["start"], e["end"], color=color, alpha=0.15)

    for _, w in warns.iterrows():
        color = "#2ca02c" if w["kind"] == "hypo" else "#d62728"
        y = 60 if w["kind"] == "hypo" else 190
        ax.scatter(w["fired_at"], y, marker="v", color=color, s=45, zorder=5)

    ax.set_title(
        f"Episodes (shaded) vs warnings (triangles) — patient {pid} — "
        f"{model.name}"
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Glucose (mg/dL)")
    fig.tight_layout()
    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
