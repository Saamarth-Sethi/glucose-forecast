"""Clinical-utility tuning of the alarm operating point.

An early-warning system has a knob: how eager to alarm. Turn it up and you
catch more events but cry wolf; turn it down and you miss events. This module
sweeps that knob — the crossing-probability threshold and the hysteresis
(minimum consecutive predictions) — and scores each setting with a utility
that trades detection against alarm fatigue::

    utility = mean_event_sensitivity - FALSE_ALARM_WEIGHT * false_alarms_per_day

It returns the full sweep, the best operating point, and a plot of the
sensitivity vs. false-alarms-per-day trade-off (an alarm ROC-like curve).
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import (
    FALSE_ALARM_WEIGHT,
    HYPER_L1,
    HYPO_L1,
    MIN_CONSECUTIVE_ALERTS,
    OUTPUT_DIR,
    PROB_THRESHOLDS,
)
from .evaluate import observation_days, score_warnings
from .warnings import generate_warnings_prob, true_events


def build_utility_curve(
    conformal,
    test_df: pd.DataFrame,
    prob_thresholds: tuple[float, ...] = PROB_THRESHOLDS,
    hysteresis_options: tuple[int, ...] = (1, 2, 3),
    false_alarm_weight: float = FALSE_ALARM_WEIGHT,
) -> tuple[pd.DataFrame, dict]:
    """Sweep alarm settings and score utility for each.

    Returns
    -------
    (sweep_df, best_row) where ``best_row`` is the utility-maximising setting.
    """
    hypo_p, hyper_p = conformal.hypo_hyper_probs(test_df, HYPO_L1, HYPER_L1)
    events = true_events(test_df)
    days = observation_days(test_df)

    rows = []
    for k in hysteresis_options:
        for p in prob_thresholds:
            warns = generate_warnings_prob(test_df, hypo_p, hyper_p, p, min_consecutive=k)
            m = score_warnings(warns, events, days)
            sens = np.nanmean(
                [m["hypo_episode_sensitivity"], m["hyper_episode_sensitivity"]]
            )
            fa = m["hypo_false_alarms_per_day"] + m["hyper_false_alarms_per_day"]
            lead = np.nanmean([m["hypo_median_lead_min"], m["hyper_median_lead_min"]])
            rows.append(
                {
                    "p_threshold": p,
                    "min_consecutive": k,
                    "mean_sensitivity": sens,
                    "false_alarms_per_day": fa,
                    "mean_lead_min": lead,
                    "utility": sens - false_alarm_weight * fa,
                }
            )

    sweep = pd.DataFrame(rows)
    best = sweep.sort_values("utility", ascending=False).iloc[0].to_dict()
    return sweep, best


def plot_utility_curve(
    sweep: pd.DataFrame,
    best: dict,
    filename: str = "utility_curve.png",
) -> str:
    """Plot sensitivity vs. false-alarms/day, coloured by utility, marking best."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Pareto-style scatter across the swept operating points.
    for k, grp in sweep.groupby("min_consecutive"):
        grp = grp.sort_values("false_alarms_per_day")
        ax.plot(
            grp["false_alarms_per_day"],
            grp["mean_sensitivity"],
            marker="o",
            ms=4,
            alpha=0.7,
            label=f"min_consecutive={k}",
        )

    ax.scatter(
        [best["false_alarms_per_day"]],
        [best["mean_sensitivity"]],
        s=180,
        facecolors="none",
        edgecolors="red",
        linewidths=2,
        zorder=5,
        label=(
            f"chosen (p={best['p_threshold']:.1f}, "
            f"k={int(best['min_consecutive'])}, U={best['utility']:.2f})"
        ),
    )

    ax.set_xlabel("False alarms per day (nuisance cost)")
    ax.set_ylabel("Mean event sensitivity (hypo & hyper)")
    ax.set_title("Alarm operating characteristic — clinical-utility trade-off")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
