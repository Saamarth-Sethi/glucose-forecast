"""Forecasting models: baselines, XGBoost (primary), and an optional LSTM.

Design choice — **one model per horizon**. Independent per-horizon models let
each learn its own error structure (a 60-min forecast is a very different
problem from a 15-min one) and keep the baselines directly comparable. The
cost (training N small models) is negligible on a laptop.

Every model exposes the same tiny interface::

    model.fit(train_df, val_df, feature_cols)   # no-op for baselines
    preds = model.predict(df, horizon_min)      # -> np.ndarray

so :mod:`evaluate` can treat them interchangeably.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import HORIZONS_MIN, SEED, horizon_steps

# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


class PersistenceModel:
    """Naive persistence: ŷ(t+PH) = y(t). The bar every model must clear."""

    name = "Persistence"

    def fit(self, *_args, **_kwargs) -> "PersistenceModel":
        return self

    def predict(self, df: pd.DataFrame, horizon_min: int) -> np.ndarray:
        return df["glucose"].to_numpy()


class LinearExtrapModel:
    """Linear extrapolation from the recent slope.

    Slope is estimated over the last 15 min (current vs. 3 steps back) and
    projected forward PH steps, then clamped to a plausible range.
    """

    name = "LinearExtrap"

    def fit(self, *_args, **_kwargs) -> "LinearExtrapModel":
        return self

    def predict(self, df: pd.DataFrame, horizon_min: int) -> np.ndarray:
        cur = df["glucose"].to_numpy()
        # lag_3 exists from feature engineering (3 steps == 15 min back).
        past = df["lag_3"].to_numpy() if "lag_3" in df else df["lag_1"].to_numpy()
        span = 3 if "lag_3" in df else 1
        slope = (cur - past) / span  # per 5-min step
        pred = cur + slope * horizon_steps(horizon_min)
        return np.clip(pred, 40.0, 400.0)


# ---------------------------------------------------------------------------
# Primary model: gradient-boosted trees (XGBoost)
# ---------------------------------------------------------------------------


def _xgboost_importable() -> bool:
    """XGBoost needs the OpenMP runtime; importing can fail on bare laptops."""
    try:
        import xgboost  # noqa: F401

        return True
    except Exception:
        return False


def make_boosted_regressor(params: dict | None = None, quantile: float | None = None):
    """Factory for a gradient-boosted regressor used by tuning and conformal.

    Uses scikit-learn's ``HistGradientBoostingRegressor`` so that quantile
    (pinball) loss and OpenMP-free operation always work. Pass ``quantile`` in
    (0, 1) to fit that conditional quantile instead of the mean.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    p = {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05}
    p.update(params or {})
    kw = dict(
        max_iter=int(p["n_estimators"]),
        max_depth=int(p["max_depth"]),
        learning_rate=float(p["learning_rate"]),
        l2_regularization=1.0,
        min_samples_leaf=20,
        random_state=SEED,
    )
    if quantile is None:
        return HistGradientBoostingRegressor(loss="squared_error", **kw)
    return HistGradientBoostingRegressor(loss="quantile", quantile=quantile, **kw)


@dataclass
class XGBModel:
    """Per-horizon gradient-boosted trees. The default deliverable.

    Uses XGBoost when it's importable, otherwise transparently falls back to
    scikit-learn's ``HistGradientBoostingRegressor`` (no system deps) so the
    project still runs on a laptop without OpenMP/``libomp``. The backend used
    is reflected in ``name``.
    """

    name: str = "XGBoost"
    feature_cols: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)  # canonical overrides from tuning
    _models: dict = field(default_factory=dict)
    _backend: str = "xgboost"

    def _canonical(self) -> dict:
        """Canonical hyperparameters (tuning overrides the defaults)."""
        base = {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.05}
        base.update(self.params or {})
        return base

    def _make_regressor(self):
        p = self._canonical()
        if _xgboost_importable():
            import xgboost as xgb

            self._backend = "xgboost"
            self.name = "XGBoost"
            return xgb.XGBRegressor(
                n_estimators=int(p["n_estimators"]),
                max_depth=int(p["max_depth"]),
                learning_rate=float(p["learning_rate"]),
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_lambda=1.0,
                objective="reg:squarederror",
                random_state=SEED,
                n_jobs=-1,
            )
        from sklearn.ensemble import HistGradientBoostingRegressor

        self._backend = "sklearn-hgb"
        self.name = "HistGBR"
        return HistGradientBoostingRegressor(
            max_iter=int(p["n_estimators"]),
            max_depth=int(p["max_depth"]),
            learning_rate=float(p["learning_rate"]),
            l2_regularization=1.0,
            min_samples_leaf=20,
            random_state=SEED,
        )

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None,
        feature_cols: list[str],
    ) -> "XGBModel":
        self.feature_cols = feature_cols
        for h in HORIZONS_MIN:
            target = f"y_h{h}"
            model = self._make_regressor()
            if self._backend == "xgboost" and val_df is not None and len(val_df) > 0:
                model.fit(
                    train_df[feature_cols],
                    train_df[target],
                    eval_set=[(val_df[feature_cols], val_df[target])],
                    verbose=False,
                )
            else:
                model.fit(train_df[feature_cols], train_df[target])
            self._models[h] = model
        return self

    def predict(self, df: pd.DataFrame, horizon_min: int) -> np.ndarray:
        model = self._models[horizon_min]
        return model.predict(df[self.feature_cols])

    def feature_importance(self, horizon_min: int) -> pd.Series | None:
        """Feature importances (XGBoost backend only; None otherwise)."""
        model = self._models[horizon_min]
        if not hasattr(model, "feature_importances_"):
            return None
        return pd.Series(
            model.feature_importances_, index=self.feature_cols
        ).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Stretch model: LSTM (optional, torch-guarded)
# ---------------------------------------------------------------------------


def torch_available() -> bool:
    """Return True if PyTorch can be imported."""
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


class LSTMModel:
    """Optional LSTM over the raw lagged glucose sequence.

    Guarded so the pipeline still runs when torch is absent — callers should
    check :func:`torch_available` first. One small LSTM is trained per horizon
    on CPU; kept intentionally light so it finishes on a laptop.
    """

    name = "LSTM"

    def __init__(self, lookback: int = 12, epochs: int = 8, hidden: int = 32):
        self.lookback = lookback
        self.epochs = epochs
        self.hidden = hidden
        self.feature_cols: list[str] = []
        self._nets: dict = {}
        self._norm: dict = {}
        self._seq_cols: list[str] = []

    # -- sequence construction --------------------------------------------
    def _seq_columns(self, df: pd.DataFrame) -> list[str]:
        # Use lag_lookback .. lag_1, then current glucose: oldest -> newest.
        cols = [f"lag_{k}" for k in range(self.lookback, 0, -1) if f"lag_{k}" in df]
        cols.append("glucose")
        return cols

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None,
        feature_cols: list[str],
    ) -> "LSTMModel":
        import torch
        from torch import nn

        torch.manual_seed(SEED)
        self.feature_cols = feature_cols
        self._seq_cols = self._seq_columns(train_df)

        mu = train_df["glucose"].mean()
        sd = train_df["glucose"].std() + 1e-6
        self._norm = {"mu": float(mu), "sd": float(sd)}

        X = ((train_df[self._seq_cols].to_numpy() - mu) / sd).astype("float32")
        X = X[:, :, None]  # (N, T, 1)

        for h in HORIZONS_MIN:
            y = ((train_df[f"y_h{h}"].to_numpy() - mu) / sd).astype("float32")
            net = _LSTMNet(hidden=self.hidden)
            opt = torch.optim.Adam(net.parameters(), lr=1e-2)
            loss_fn = nn.MSELoss()
            Xt = torch.from_numpy(X)
            yt = torch.from_numpy(y)[:, None]
            n = len(Xt)
            batch = 256
            net.train()
            for _ in range(self.epochs):
                perm = torch.randperm(n)
                for i in range(0, n, batch):
                    idx = perm[i : i + batch]
                    opt.zero_grad()
                    out = net(Xt[idx])
                    loss = loss_fn(out, yt[idx])
                    loss.backward()
                    opt.step()
            net.eval()
            self._nets[h] = net
        return self

    def predict(self, df: pd.DataFrame, horizon_min: int) -> np.ndarray:
        import torch

        mu, sd = self._norm["mu"], self._norm["sd"]
        X = ((df[self._seq_cols].to_numpy() - mu) / sd).astype("float32")[:, :, None]
        with torch.no_grad():
            out = self._nets[horizon_min](torch.from_numpy(X)).numpy().ravel()
        return out * sd + mu


def _make_lstm_net(hidden: int):
    from torch import nn

    class _Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):  # x: (N, T, 1)
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

    return _Net()


# Lazy factory so importing this module never requires torch.
def _LSTMNet(hidden: int):  # noqa: N802 - kept as a pseudo-class factory
    return _make_lstm_net(hidden)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def build_models(include_lstm: bool = True) -> list:
    """Instantiate the model line-up, skipping LSTM if torch is missing."""
    models = [PersistenceModel(), LinearExtrapModel(), XGBModel()]
    if include_lstm and torch_available():
        models.append(LSTMModel())
    return models
