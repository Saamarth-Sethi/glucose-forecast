"""Central configuration and clinical constants.

All tunable knobs live here so the rest of the code reads cleanly and the
whole pipeline is reproducible from a single place.
"""
from __future__ import annotations

# --- Reproducibility -------------------------------------------------------
SEED: int = 42

# --- CGM sampling ----------------------------------------------------------
SAMPLE_MINUTES: int = 5  # CGM samples interstitial glucose ~every 5 min.

# --- Prediction horizons ---------------------------------------------------
# Minutes ahead to forecast. At 5-min sampling these are 3, 6, 12 steps.
HORIZONS_MIN: tuple[int, ...] = (15, 30, 60)


def horizon_steps(horizon_min: int) -> int:
    """Convert a horizon in minutes to a number of 5-min steps."""
    return horizon_min // SAMPLE_MINUTES


# --- Clinical thresholds (mg/dL) ------------------------------------------
HYPO_L1: float = 70.0   # hypoglycemia, level 1
HYPO_L2: float = 54.0   # hypoglycemia, level 2 (clinically significant)
HYPER_L1: float = 180.0  # hyperglycemia, postprandial
HYPER_L2: float = 250.0  # hyperglycemia, severe

# Physiologically plausible clamp range for CGM readings.
GLUCOSE_MIN: float = 40.0
GLUCOSE_MAX: float = 400.0

# --- Gap handling ----------------------------------------------------------
# Interpolate gaps up to this length; anything longer breaks a segment and we
# never train or predict across it.
MAX_INTERP_GAP_MIN: int = 15

# --- Feature engineering ---------------------------------------------------
# How far back (minutes) to build lag / rolling features.
LOOKBACK_MIN: int = 60

# Exogenous "on-board" decay time constants (minutes) for causal accumulation.
IOB_TAU_MIN: float = 120.0  # insulin-on-board decay
COB_TAU_MIN: float = 60.0   # carbs-on-board decay
ACT_TAU_MIN: float = 45.0   # activity after-effect decay

# --- Early-warning logic ---------------------------------------------------
# Require this many consecutive threshold-crossing predictions before firing,
# to suppress flickering alarms (hysteresis).
MIN_CONSECUTIVE_ALERTS: int = 2

# --- Synthetic data defaults ----------------------------------------------
N_PATIENTS: int = 10
N_DAYS: int = 7
N_TEST_PATIENTS: int = 3  # entire patients held out for the test set.

# --- Conformal / probabilistic forecasting ---------------------------------
# Miscoverage rate: 0.2 -> 80% prediction intervals.
CONFORMAL_ALPHA: float = 0.2
# Quantile grid fit for the predictive CDF (enables P(hypo)/P(hyper)).
QUANTILE_GRID: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)

# --- Hyperparameter tuning (nested, time-series CV) ------------------------
N_TSCV_FOLDS: int = 3  # expanding-window folds inside the dev set.

# --- Per-patient calibration ----------------------------------------------
# Fraction of each test patient's earliest data used to calibrate to them.
CALIB_FRACTION: float = 0.2

# --- Clinical-utility trade-off -------------------------------------------
# Utility = sensitivity - FALSE_ALARM_WEIGHT * (false alarms per day).
# Encodes alarm fatigue: how many missed-event-equivalents one nuisance
# alarm/day costs. Tune to the care setting.
FALSE_ALARM_WEIGHT: float = 0.10
# Probability thresholds swept when building the utility curve.
PROB_THRESHOLDS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)

# --- Output ----------------------------------------------------------------
OUTPUT_DIR: str = "outputs"
