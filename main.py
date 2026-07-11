"""End-to-end glucose forecasting + early-warning pipeline.

Run ``python main.py`` to execute everything on synthetic data with zero
external inputs: generate -> preprocess -> causal features -> nested
time-series hyperparameter tuning -> train baselines + XGBoost (+ optional
LSTM) -> evaluate per horizon -> conformal prediction intervals + crossing
probabilities -> per-patient calibration -> early-warning system ->
clinical-utility tuning of the alarm operating point. Prints tables and saves
plots to ``outputs/``.

CLI (all optional)::

    python main.py                          # full pipeline on builtin synthetic data
    python main.py --source simglucose --patients 4 --days 3
    python main.py --csv data.csv --ts-col time --glucose-col mg_dl
    python main.py --fast                   # skip tuning/conformal/calibration/utility
    python main.py --no-lstm --no-tune
"""
from __future__ import annotations

import argparse
import os
import random
import warnings as _warnings

import numpy as np
import pandas as pd

from data.generate import generate_dataset
from src import load
from src.conformal import ConformalForecaster, interval_metrics
from src.config import (
    HORIZONS_MIN,
    HYPER_L1,
    HYPO_L1,
    N_DAYS,
    N_PATIENTS,
    N_TEST_PATIENTS,
    OUTPUT_DIR,
    SEED,
)
from src.evaluate import (
    evaluate_model,
    mae,
    plot_clarke_grid,
    plot_forecast_intervals,
    plot_prediction_overlay,
    plot_warning_timeline,
    predict_all_horizons,
    rmse,
    warning_system_metrics,
)
from src.features import build_features, feature_columns
from src.models import LinearExtrapModel, LSTMModel, PersistenceModel, XGBModel, torch_available
from src.preprocess import preprocess
from src.tuning import PerPatientCalibrator, tune_hyperparameters
from src.utility import build_utility_curve, plot_utility_curve

_warnings.filterwarnings("ignore", category=FutureWarning)
_warnings.filterwarnings("ignore", category=UserWarning)

HEADLINE_HORIZON = 30  # horizon used for tuning selection and headline plots.


def set_seed(seed: int = SEED) -> None:
    """Fix all RNGs we touch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def split_data(feat: pd.DataFrame):
    """Leakage-free split (hold out whole patients, or per-patient time split)."""
    patients = sorted(feat["patient_id"].unique())
    if len(patients) >= 4:
        n_test = min(N_TEST_PATIENTS, max(1, len(patients) // 3))
        test_patients = set(patients[-n_test:])
        test = feat[feat["patient_id"].isin(test_patients)].copy()
        trainval = feat[~feat["patient_id"].isin(test_patients)].copy()
        cutoff = trainval["timestamp"].quantile(0.8)
        train = trainval[trainval["timestamp"] <= cutoff].copy()
        val = trainval[trainval["timestamp"] > cutoff].copy()
        return train, val, test, sorted(test_patients)

    train_parts, val_parts, test_parts = [], [], []
    for _pid, sub in feat.groupby("patient_id"):
        sub = sub.sort_values("timestamp")
        t70, t85 = sub["timestamp"].quantile(0.70), sub["timestamp"].quantile(0.85)
        train_parts.append(sub[sub["timestamp"] <= t70])
        val_parts.append(sub[(sub["timestamp"] > t70) & (sub["timestamp"] <= t85)])
        test_parts.append(sub[sub["timestamp"] > t85])
    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(val_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
        patients,
    )


def print_table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> None:
    """Compact fixed-width table printer (no external deps)."""
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: floatfmt.format(v) if pd.notna(v) else "nan")
    widths = {c: max(len(str(c)), df[c].astype(str).map(len).max()) for c in df.columns}
    header = "  ".join(str(c).rjust(widths[c]) for c in df.columns)
    print(header)
    print("-" * len(header))
    for _, row in df.iterrows():
        print("  ".join(str(row[c]).rjust(widths[c]) for c in df.columns))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Glucose forecasting pipeline")
    p.add_argument("--csv", type=str, default=None, help="Path to a real CGM CSV")
    p.add_argument("--ts-col", type=str, default="timestamp")
    p.add_argument("--glucose-col", type=str, default="glucose")
    p.add_argument("--patient-col", type=str, default=None)
    p.add_argument("--source", choices=["builtin", "simglucose", "auto"], default="builtin")
    p.add_argument("--patients", type=int, default=N_PATIENTS)
    p.add_argument("--days", type=int, default=N_DAYS)
    p.add_argument("--no-lstm", action="store_true")
    p.add_argument("--no-tune", action="store_true", help="Skip hyperparameter tuning")
    p.add_argument("--no-conformal", action="store_true", help="Skip conformal intervals")
    p.add_argument("--no-calibrate", action="store_true", help="Skip per-patient calibration")
    p.add_argument("--no-utility", action="store_true", help="Skip clinical-utility sweep")
    p.add_argument("--fast", action="store_true", help="Skip all optional heavy stages")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.fast:
        args.no_tune = args.no_conformal = args.no_calibrate = args.no_utility = True
    set_seed()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1. Data ----------------------------------------------------------
    if args.csv:
        print(f"[data] Loading real CSV: {args.csv}")
        raw = load.load_csv(
            args.csv, timestamp_col=args.ts_col, glucose_col=args.glucose_col,
            patient_col=args.patient_col,
        )
        source = "csv"
    else:
        raw_df, source = generate_dataset(
            n_patients=args.patients, n_days=args.days, seed=SEED, source=args.source
        )
        raw = load.from_frame(raw_df)
    print(f"[data] source={source}  rows={len(raw):,}  patients={raw['patient_id'].nunique()}")

    # --- 2. Preprocess + 3. Features -------------------------------------
    proc = preprocess(raw)
    print(f"[preprocess] rows={len(proc):,}  segments={proc['segment'].nunique()}  "
          f"imputed={int(proc['imputed'].sum())}")
    feat = build_features(proc)
    fcols = feature_columns(feat)
    print(f"[features] samples={len(feat):,}  n_features={len(fcols)}  "
          f"(incl. IOB/COB/activity exogenous features)")

    # --- 4. Split ---------------------------------------------------------
    train, val, test, test_patients = split_data(feat)
    dev = pd.concat([train, val], ignore_index=True)
    print(f"[split] train={len(train):,}  val={len(val):,}  test={len(test):,}  "
          f"held-out test patients={test_patients}")

    # --- 5. Nested time-series hyperparameter tuning ----------------------
    best_params: dict = {}
    if not args.no_tune:
        best_params, cv_table = tune_hyperparameters(dev, fcols, HEADLINE_HORIZON)
        print(f"\n===== NESTED TIME-SERIES CV TUNING (inner loop, +{HEADLINE_HORIZON} min) =====")
        print_table(cv_table, floatfmt="{:.3f}")
        print(f"[tune] selected: {best_params}")

    # --- 6. Train models --------------------------------------------------
    models = [PersistenceModel(), LinearExtrapModel(), XGBModel(params=best_params)]
    if not args.no_lstm and torch_available():
        models.append(LSTMModel())
    print(f"\n[train] models: {[m.name for m in models]}")
    for m in models:
        m.fit(train, val, fcols)
    best = next(m for m in models if isinstance(m, XGBModel))

    # --- 7. Evaluate ------------------------------------------------------
    per_model = pd.concat([evaluate_model(m, test) for m in models], ignore_index=True)
    per_model.to_csv(os.path.join(OUTPUT_DIR, "metrics_full.csv"), index=False)

    print("\n================ REGRESSION + CLARKE (test set) ================")
    print_table(per_model[["model", "horizon_min", "RMSE", "MAE", "ClarkeA%", "ClarkeA+B%"]])
    print("\n================ EVENT DETECTION — HYPO (<70), sample-level ================")
    print_table(per_model[["model", "horizon_min", "hypo_sens", "hypo_spec", "hypo_prec", "hypo_f1"]], "{:.3f}")
    print("\n================ EVENT DETECTION — HYPER (>180), sample-level ================")
    print_table(per_model[["model", "horizon_min", "hyper_sens", "hyper_spec", "hyper_prec", "hyper_f1"]], "{:.3f}")

    print("\n================ RMSE BY HORIZON (mg/dL, lower is better) ================")
    pivot = per_model.pivot(index="model", columns="horizon_min", values="RMSE")
    pivot.columns = [f"+{c}min" for c in pivot.columns]
    print_table(pivot.reset_index())

    # --- 8. Conformal prediction intervals + crossing probabilities -------
    conformal = None
    if not args.no_conformal:
        print("\n[conformal] fitting quantile models + CQR calibration ...")
        conformal = ConformalForecaster(params=best_params).fit(train, fcols).calibrate(val)
        cov_rows = []
        for h in HORIZONS_MIN:
            lo, hi = conformal.predict_interval(test, h)
            im = interval_metrics(lo, hi, test[f"y_h{h}"].to_numpy())
            cov_rows.append({"horizon_min": h, "target_cov": 1 - conformal.alpha,
                             "empirical_cov": im["coverage"], "mean_width_mgdl": im["mean_width"]})
        print("======= CONFORMAL 80% INTERVAL COVERAGE (test set) =======")
        print_table(pd.DataFrame(cov_rows), "{:.3f}")

    # --- 9. Per-patient calibration --------------------------------------
    if not args.no_calibrate:
        calibrator = PerPatientCalibrator()
        eval_df, raw_preds, cal_preds = calibrator.fit_apply(best, test)
        rows = []
        for h in HORIZONS_MIN:
            yt = eval_df[f"y_h{h}"].to_numpy()
            rows.append({
                "horizon_min": h,
                "RMSE_raw": rmse(yt, raw_preds[h]), "RMSE_calib": rmse(yt, cal_preds[h]),
                "MAE_raw": mae(yt, raw_preds[h]), "MAE_calib": mae(yt, cal_preds[h]),
            })
        cal_tbl = pd.DataFrame(rows)
        cal_tbl["RMSE_gain%"] = 100 * (cal_tbl["RMSE_raw"] - cal_tbl["RMSE_calib"]) / cal_tbl["RMSE_raw"]
        kept = sum(1 for ab in calibrator._coef.values() if ab != (1.0, 0.0))
        print("\n===== PER-PATIENT CALIBRATION (validation-gated, adapt to early data) =====")
        print_table(cal_tbl, "{:.2f}")
        print(f"[calibrate] gate kept a correction for {kept}/{len(calibrator._coef)} "
              f"patient-horizons; near-neutral because autoregressive features "
              f"already capture per-patient level.")

    # --- 10. Early-warning system (deterministic) ------------------------
    preds = predict_all_horizons(best, test)
    ws = warning_system_metrics(test, preds)
    print(f"\n========= EARLY-WARNING SYSTEM ({best.name}, deterministic) =========")
    ws_tbl = pd.DataFrame([
        {"kind": k, "episodes": int(ws[f"{k}_episodes"]), "detected": int(ws[f"{k}_detected"]),
         "sensitivity": ws[f"{k}_episode_sensitivity"], "median_lead_min": ws[f"{k}_median_lead_min"],
         "false_alarms_per_day": ws[f"{k}_false_alarms_per_day"]}
        for k in ("hypo", "hyper")
    ])
    print_table(ws_tbl, "{:.2f}")

    # --- 11. Clinical-utility tuning of the alarm operating point ----------
    best_op = None
    if not args.no_utility and conformal is not None:
        print("\n[utility] sweeping probability threshold x hysteresis ...")
        sweep, best_op = build_utility_curve(conformal, test)
        sweep.to_csv(os.path.join(OUTPUT_DIR, "utility_sweep.csv"), index=False)
        print("===== CLINICAL-UTILITY: best alarm operating point =====")
        print(f"  P(cross) threshold = {best_op['p_threshold']:.2f}, "
              f"min_consecutive = {int(best_op['min_consecutive'])}")
        print(f"  mean sensitivity = {best_op['mean_sensitivity']:.2f}, "
              f"false alarms/day = {best_op['false_alarms_per_day']:.2f}, "
              f"mean lead = {best_op['mean_lead_min']:.1f} min, utility = {best_op['utility']:.3f}")

    # --- 12. Plots --------------------------------------------------------
    plots = [
        plot_prediction_overlay(best, test, horizon_min=HEADLINE_HORIZON),
        plot_clarke_grid(test[f"y_h{HEADLINE_HORIZON}"].to_numpy(),
                         best.predict(test, HEADLINE_HORIZON),
                         title=f"Clarke Error Grid — {best.name} +{HEADLINE_HORIZON} min"),
        plot_warning_timeline(best, test),
    ]
    if conformal is not None:
        plots.append(plot_forecast_intervals(conformal, test, horizon_min=HEADLINE_HORIZON))
    if best_op is not None:
        plots.append(plot_utility_curve(sweep, best_op))
    print("\n[plots] saved:")
    for p in plots:
        print(f"        {p}")
    print("\nDone. Metrics -> outputs/metrics_full.csv")


if __name__ == "__main__":
    main()
