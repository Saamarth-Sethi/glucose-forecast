"""Data loading and the common data interface.

Everything downstream consumes a single canonical long-format schema::

    patient_id : int/str   grouping key (one virtual or real subject)
    timestamp  : tz-aware datetime, monotonic per patient
    glucose    : float, mg/dL
    carbs      : float, grams of carbohydrate at that step   (0 if unknown)
    insulin    : float, units of insulin at that step        (0 if unknown)
    activity   : float, 0..1 exercise intensity              (0 if unknown)

Three entry points return this schema: the synthetic generator (via
:func:`from_frame`), a generic CSV loader, and an OhioT1DM XML loader.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("timestamp", "glucose")
EXOG_COLUMNS = ("carbs", "insulin", "activity")


def load_csv(
    path: str | Path,
    timestamp_col: str = "timestamp",
    glucose_col: str = "glucose",
    patient_col: str | None = None,
    carbs_col: str | None = None,
    insulin_col: str | None = None,
    activity_col: str | None = None,
    tz: str = "UTC",
) -> pd.DataFrame:
    """Load a real CGM export into the canonical schema.

    Glucose is assumed to be **mg/dL**; if yours is mmol/L, multiply by 18.0.
    Exogenous columns (carbs/insulin/activity) are optional — pass their names
    if present, otherwise they default to 0.

    CUSTOMIZE example for a two-column export::

        load_csv("mydata.csv", timestamp_col="time", glucose_col="mg_dl")
    """
    df = pd.read_csv(path)
    if timestamp_col not in df.columns or glucose_col not in df.columns:
        raise ValueError(
            f"CSV must contain '{timestamp_col}' and '{glucose_col}'. "
            f"Found: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df[timestamp_col], utc=False, errors="coerce")
    out["glucose"] = pd.to_numeric(df[glucose_col], errors="coerce")
    out["patient_id"] = (
        df[patient_col].values if patient_col and patient_col in df.columns else 0
    )
    for canon, src in (
        ("carbs", carbs_col),
        ("insulin", insulin_col),
        ("activity", activity_col),
    ):
        out[canon] = (
            pd.to_numeric(df[src], errors="coerce").fillna(0.0)
            if src and src in df.columns
            else 0.0
        )
    return _finalize(out, tz=tz)


def load_ohio_t1dm(path: str | Path, tz: str = "UTC") -> pd.DataFrame:
    """Load an OhioT1DM ``.xml`` record into the canonical schema.

    The OhioT1DM dataset (Marling & Bunescu) ships one XML file per patient
    with ``<glucose_level>``, ``<meal>`` (carbs), ``<bolus>``/``<basal>``
    (insulin) and ``<exercise>`` event lists. It requires a data-use
    agreement, so it is not bundled here — point this at your own copy.

    Notes
    -----
    Events (meals, boluses, exercise) are sparse timestamps; they are aligned
    onto the glucose timeline by nearest 5-min bin during preprocessing.
    """
    import xml.etree.ElementTree as ET

    root = ET.parse(str(path)).getroot()
    pid = root.attrib.get("id", Path(path).stem)

    def _events(tag: str, value_attr: str, time_attr: str = "ts") -> pd.DataFrame:
        node = root.find(tag)
        rows = []
        if node is not None:
            for ev in node.findall("event"):
                ts = ev.attrib.get(time_attr) or ev.attrib.get("ts_begin")
                val = ev.attrib.get(value_attr)
                if ts is not None and val is not None:
                    rows.append((ts, float(val)))
        return pd.DataFrame(rows, columns=["timestamp", "value"])

    glu = _events("glucose_level", "value")
    if glu.empty:
        raise ValueError(f"No <glucose_level> events found in {path}")
    glu["timestamp"] = pd.to_datetime(glu["timestamp"], errors="coerce", dayfirst=False)

    out = pd.DataFrame(
        {
            "patient_id": pid,
            "timestamp": glu["timestamp"],
            "glucose": glu["value"],
            "carbs": 0.0,
            "insulin": 0.0,
            "activity": 0.0,
        }
    ).dropna(subset=["timestamp"])

    # Merge sparse events onto the nearest existing glucose timestamp.
    def _merge(evt: pd.DataFrame, col: str) -> None:
        if evt.empty:
            return
        evt = evt.copy()
        evt["timestamp"] = pd.to_datetime(evt["timestamp"], errors="coerce")
        evt = evt.dropna(subset=["timestamp"])
        idx = out["timestamp"].searchsorted(evt["timestamp"])
        idx = np.clip(idx, 0, len(out) - 1)
        for i, v in zip(idx, evt["value"].to_numpy()):
            out.iat[i, out.columns.get_loc(col)] += v

    _merge(_events("meal", "carbs"), "carbs")
    _merge(_events("bolus", "dose", time_attr="ts_begin"), "insulin")
    return _finalize(out, tz=tz)


def from_frame(df: pd.DataFrame, tz: str = "UTC") -> pd.DataFrame:
    """Coerce an in-memory frame (e.g. the synthetic generator) to canon."""
    return _finalize(df.copy(), tz=tz)


def _finalize(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Enforce schema invariants: tz-aware, monotonic, de-duplicated."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "patient_id" not in df.columns:
        df["patient_id"] = 0
    for col in EXOG_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    ts = pd.to_datetime(df["timestamp"])
    ts = ts.dt.tz_localize(tz) if ts.dt.tz is None else ts.dt.tz_convert(tz)
    df["timestamp"] = ts

    df = df.dropna(subset=["timestamp"])
    df = df.sort_values(["patient_id", "timestamp"])
    df = df.drop_duplicates(subset=["patient_id", "timestamp"], keep="last")
    df = df.reset_index(drop=True)

    return df[["patient_id", "timestamp", "glucose", *EXOG_COLUMNS]]
