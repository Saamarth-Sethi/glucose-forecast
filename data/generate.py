"""Synthetic CGM data generation with rich exogenous inputs.

Two backends behind one function, :func:`generate_dataset`:

1. ``simglucose`` (UVA/Padova) — the validated T1D simulator. It natively
   emits carbohydrate intake (CHO) and insulin delivery, which we carry
   through as exogenous inputs. Opt in with ``source="simglucose"``.
2. A self-contained physiological generator (default) that models the same
   channels explicitly: meals -> carbohydrate appearance, insulin boluses +
   basal, and exercise/activity bouts, each with its own effect curve on
   glucose. This lets the downstream feature layer compute insulin-on-board
   (IOB) and carbs-on-board (COB), so the exogenous inputs genuinely help.

Both backends return a tidy long DataFrame with the canonical schema::

    patient_id, timestamp, glucose, carbs, insulin, activity

``carbs`` are grams per 5-min step, ``insulin`` units per step, ``activity`` a
0..1 normalized exercise intensity (simglucose has no activity channel, so it
is 0 there).
"""
from __future__ import annotations

import contextlib
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import (  # noqa: E402
    GLUCOSE_MAX,
    GLUCOSE_MIN,
    N_DAYS,
    N_PATIENTS,
    SAMPLE_MINUTES,
    SEED,
)

# ---------------------------------------------------------------------------
# simglucose backend (validated UVA/Padova simulator)
# ---------------------------------------------------------------------------


def _try_simglucose(n_patients: int, n_days: int, seed: int) -> pd.DataFrame | None:
    """Generate data with simglucose. Return None if unavailable/broken.

    Runs one basal-bolus-controlled simulation per virtual adult patient and
    keeps CGM (glucose), CHO (carbs) and insulin. Progress output is silenced.
    """
    try:
        from datetime import datetime, timedelta

        from simglucose.actuator.pump import InsulinPump
        from simglucose.controller.basal_bolus_ctrller import BBController
        from simglucose.patient.t1dpatient import T1DPatient
        from simglucose.sensor.cgm import CGMSensor
        from simglucose.simulation.env import T1DSimEnv
        from simglucose.simulation.scenario_gen import RandomScenario
        from simglucose.simulation.sim_engine import SimObj, sim
    except Exception:
        return None

    try:
        logging.disable(logging.CRITICAL)
        start = datetime(2024, 1, 1, 0, 0, 0)
        frames = []
        for pid in range(n_patients):
            name = f"adult#{(pid % 10) + 1:03d}"
            patient = T1DPatient.withName(name)
            sensor = CGMSensor.withName("Dexcom", seed=seed + pid)
            pump = InsulinPump.withName("Insulet")
            scenario = RandomScenario(start_time=start, seed=seed + pid)
            env = T1DSimEnv(patient, sensor, pump, scenario)
            with tempfile.TemporaryDirectory() as d:
                sim_obj = SimObj(
                    env, BBController(), timedelta(days=n_days), animate=False, path=d
                )
                with open(os.devnull, "w") as fn, contextlib.redirect_stdout(
                    fn
                ), contextlib.redirect_stderr(fn):
                    res = sim(sim_obj)
            df = res.reset_index().rename(columns={"Time": "timestamp", "CGM": "glucose"})
            frames.append(
                pd.DataFrame(
                    {
                        "patient_id": pid,
                        "timestamp": df["timestamp"],
                        "glucose": df["glucose"].clip(GLUCOSE_MIN, GLUCOSE_MAX),
                        "carbs": df.get("CHO", 0.0),
                        "insulin": df.get("insulin", 0.0),
                        "activity": 0.0,
                    }
                )
            )
        logging.disable(logging.NOTSET)
        return pd.concat(frames, ignore_index=True)
    except Exception:
        logging.disable(logging.NOTSET)
        return None


# ---------------------------------------------------------------------------
# Built-in physiological generator
# ---------------------------------------------------------------------------


def _circadian_baseline(minute_of_day: np.ndarray) -> np.ndarray:
    """A gentle 24h rhythm with a dawn phenomenon."""
    t = 2 * np.pi * minute_of_day / (24 * 60)
    return 110.0 + 12.0 * np.sin(t - np.pi / 2) + 8.0 * np.sin(2 * t)


def _decay_curve(dt: np.ndarray, rise_tau: float, decay_tau: float) -> np.ndarray:
    """Normalized rise-then-decay impulse response, 0 before the event."""
    dt = np.clip(dt, 0, None)
    return (1 - np.exp(-dt / rise_tau)) * np.exp(-dt / decay_tau)


def _synthetic_patient(pid: int, n_days: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate one virtual patient with explicit carbs/insulin/activity.

    Glucose is assembled as::

        baseline + drift + carb_effect - insulin_effect - activity_effect + noise

    and the driving channels (carbs, insulin, activity) are returned so the
    feature layer can derive IOB / COB / activity features causally.
    """
    step = SAMPLE_MINUTES
    n = (24 * 60 // step) * n_days
    times = np.arange(n) * step
    minute_of_day = times % (24 * 60)

    # Per-patient physiology (clinically-scaled).
    patient_offset = rng.normal(0, 12)
    isf = rng.uniform(35, 55)              # insulin sensitivity factor, mg/dL per unit
    carb_ratio = rng.uniform(9, 13)        # g carbs per unit insulin
    carb_factor = isf / carb_ratio * rng.uniform(0.9, 1.1)  # mg/dL per gram
    ex_sens = rng.uniform(0.7, 1.4)        # exercise sensitivity

    # Baseline shifted up to a realistic T1D operating point (median ~150).
    baseline = _circadian_baseline(minute_of_day) + patient_offset + 40.0
    drift = np.cumsum(rng.normal(0, 1.1, size=n))
    drift = (drift - np.linspace(0, drift[-1], n)) * 0.9

    carbs = np.zeros(n)
    insulin = np.zeros(n)
    activity = np.zeros(n)
    carb_effect = np.zeros(n)
    insulin_effect = np.zeros(n)
    activity_effect = np.zeros(n)

    # Basal insulin drip (steady state folded into baseline; recorded as a
    # small per-step delivery for realism / IOB features).
    basal_per_step = rng.uniform(0.01, 0.03)
    insulin += basal_per_step

    meal_anchors = [7 * 60, 12 * 60 + 30, 18 * 60 + 30, 21 * 60]
    for day in range(n_days):
        # --- meals: carbs (fast rise) + bolus (slower, mis-dosed) ---
        # Carbs act faster than insulin, so even a matched meal gives a
        # transient rise; dose errors push the net toward hyper/hypo.
        n_meals = int(rng.integers(3, 5))
        for a in rng.choice(len(meal_anchors), size=n_meals, replace=False):
            m = day * 24 * 60 + meal_anchors[a] + rng.normal(0, 20)
            grams = float(np.clip(rng.lognormal(np.log(50), 0.4), 10, 120))
            m_idx = int(np.clip(round(m / step), 0, n - 1))
            carbs[m_idx] += grams
            carb_effect += (
                carb_factor
                * grams
                * _decay_curve(times - m, rng.uniform(18, 28), rng.uniform(70, 120))
            )
            dose_err = rng.uniform(0.75, 1.2)  # <1 under-bolus, >1 over-bolus
            units = grams / carb_ratio * dose_err
            b_idx = int(np.clip(m_idx + rng.integers(0, 3), 0, n - 1))
            insulin[b_idx] += units
            # 0.9 scale + moderate decay keeps a matched meal roughly neutral
            # (carbs act first, insulin catches up) rather than net-hypo.
            insulin_effect += (
                0.9
                * isf
                * units
                * _decay_curve(times - times[b_idx], rng.uniform(35, 50), rng.uniform(90, 150))
            )

        # --- correction boluses (occasional, can overshoot to hypo) ---
        if rng.random() < 0.5:
            c = day * 24 * 60 + rng.uniform(9 * 60, 22 * 60)
            units = rng.uniform(1.0, 3.0)
            c_idx = int(np.clip(round(c / step), 0, n - 1))
            insulin[c_idx] += units
            insulin_effect += (
                isf
                * units
                * _decay_curve(times - c, rng.uniform(30, 45), rng.uniform(120, 200))
            )

        # --- exercise/activity bouts (glucose-lowering) ---
        for _ in range(int(rng.integers(0, 3))):
            e = day * 24 * 60 + rng.uniform(6 * 60, 21 * 60)
            dur = rng.uniform(30, 75)          # minutes
            intensity = rng.uniform(0.4, 1.0)
            in_bout = (times >= e) & (times <= e + dur)
            activity[in_bout] = np.maximum(activity[in_bout], intensity)
            activity_effect += (
                ex_sens
                * intensity
                * 45.0
                * _decay_curve(times - e, 12, rng.uniform(60, 120))
            )

    signal = baseline + drift + carb_effect - insulin_effect - activity_effect

    # AR(1) coloured sensor noise.
    white = rng.normal(0, 3.0, size=n)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.6 * noise[i - 1] + white[i]

    # Per-patient sensor calibration bias/gain: real CGM sensors carry a
    # systematic, subject-specific offset and scale (part of what MARD
    # captures). A population model can't know an individual's bias, which is
    # exactly what per-patient calibration corrects.
    sensor_gain = rng.normal(1.0, 0.04)
    sensor_offset = rng.normal(0.0, 8.0)
    glucose = sensor_gain * (signal + noise) + sensor_offset
    glucose = np.clip(glucose, GLUCOSE_MIN, GLUCOSE_MAX)

    start = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    ts = start + pd.to_timedelta(times, unit="m")
    df = pd.DataFrame(
        {
            "patient_id": pid,
            "timestamp": ts,
            "glucose": glucose,
            "carbs": carbs,
            "insulin": insulin,
            "activity": activity,
        }
    )

    # Occasional sensor dropout gaps (glucose NaN) to exercise gap handling.
    for _ in range(int(rng.integers(1, 3))):
        gap_len = int(rng.integers(2, 9))
        gstart = int(rng.integers(0, max(1, n - gap_len)))
        df.loc[gstart : gstart + gap_len, "glucose"] = np.nan

    return df


def generate_synthetic(
    n_patients: int = N_PATIENTS, n_days: int = N_DAYS, seed: int = SEED
) -> pd.DataFrame:
    """Built-in physiological CGM generator (no external deps)."""
    frames = [
        _synthetic_patient(pid, n_days, np.random.default_rng(seed + pid))
        for pid in range(n_patients)
    ]
    return pd.concat(frames, ignore_index=True)


def generate_dataset(
    n_patients: int = N_PATIENTS,
    n_days: int = N_DAYS,
    seed: int = SEED,
    source: str = "builtin",
) -> tuple[pd.DataFrame, str]:
    """Generate a synthetic CGM dataset.

    Parameters
    ----------
    source : {"builtin", "simglucose", "auto"}
        ``builtin`` (default) uses the fast physiological generator.
        ``simglucose`` uses the validated UVA/Padova simulator (slower).
        ``auto`` tries simglucose and falls back to builtin.

    Returns
    -------
    (df, source_used)
    """
    if source in ("simglucose", "auto"):
        df = _try_simglucose(n_patients, n_days, seed)
        if df is not None and len(df) > 0:
            return df, "simglucose"
        if source == "simglucose":
            print("[generate] simglucose unavailable; falling back to builtin.")
    return generate_synthetic(n_patients, n_days, seed), "builtin"


if __name__ == "__main__":
    data, src = generate_dataset()
    print(f"Generated {len(data):,} rows from '{src}' across "
          f"{data['patient_id'].nunique()} patients.")
    print(data.head())
