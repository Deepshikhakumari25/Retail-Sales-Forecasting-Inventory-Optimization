"""Baseline and machine-learning demand forecasters."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .features import FEATURE_COLUMNS, add_demand_features
from .intermittent import croston_sba


def seasonal_naive(values: list[float] | np.ndarray, horizon: int, season: int = 7) -> np.ndarray:
    """Repeat the latest seasonal cycle; use the mean if history is too short."""
    history = np.asarray(values, dtype=float)
    if horizon < 1 or season < 1:
        raise ValueError("horizon and season must be positive")
    if history.size == 0:
        raise ValueError("history must contain at least one observation")
    if not np.isfinite(history).all() or (history < 0).any():
        raise ValueError("history must contain finite non-negative demand")
    pattern = history[-season:] if len(history) >= season else np.repeat(history.mean(), season)
    return np.maximum(0, np.resize(pattern, horizon)).astype(float)


def _fit_forest(data: pd.DataFrame) -> RandomForestRegressor:
    featured = add_demand_features(data).dropna(subset=["lag_1", "lag_7", "lag_14"])
    if len(featured) < 10:
        raise ValueError("At least 10 rows with 14 days of history are required for Random Forest")
    model = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(featured[list(FEATURE_COLUMNS)], featured["qty_sold"])
    return model


def recursive_random_forest(
    history: pd.DataFrame,
    horizon: int,
    *,
    model: RandomForestRegressor | None = None,
) -> np.ndarray:
    """Forecast recursively, updating lags only with prior observations/predictions."""
    if horizon < 1 or history.empty:
        raise ValueError("history and a positive horizon are required")
    ordered = history.sort_values("date").copy()
    if model is None:
        model = _fit_forest(ordered)
    quantities = ordered["qty_sold"].astype(float).tolist()
    dates = pd.to_datetime(ordered["date"])
    last = dates.max()
    store_code = int(pd.Categorical(ordered["store_id"]).codes[-1])
    item_code = int(pd.Categorical(ordered["item_id"]).codes[-1])
    promo = int(ordered["promo_flag"].iloc[-1]) if "promo_flag" in ordered else 0
    output: list[float] = []
    for step in range(1, horizon + 1):
        future_date = last + pd.Timedelta(days=step)
        row = {
            "day_of_week": future_date.dayofweek,
            "day_of_month": future_date.day,
            "week_of_year": int(future_date.isocalendar().week),
            "month": future_date.month,
            "is_weekend": int(future_date.dayofweek >= 5),
            "lag_1": quantities[-1],
            "lag_7": quantities[-7] if len(quantities) >= 7 else float(np.mean(quantities)),
            "lag_14": quantities[-14] if len(quantities) >= 14 else float(np.mean(quantities)),
            "rolling_mean_7": float(np.mean(quantities[-7:])),
            "rolling_mean_14": float(np.mean(quantities[-14:])),
            "rolling_mean_28": float(np.mean(quantities[-28:])),
            "rolling_std_7": float(np.std(quantities[-7:], ddof=1)) if len(quantities[-7:]) > 1 else 0.0,
            "promo_flag": promo, "store_code": store_code, "item_code": item_code,
        }
        value = max(0.0, float(model.predict(pd.DataFrame([row])[list(FEATURE_COLUMNS)])[0]))
        output.append(value)
        quantities.append(value)
    return np.asarray(output)


def forecast_series(
    history: pd.DataFrame, horizon: int, model_name: str = "seasonal_naive"
) -> tuple[np.ndarray, str]:
    """Forecast a single series, using clear fallbacks for sparse histories."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    ordered = history.sort_values("date")
    values = ordered["qty_sold"].to_numpy(dtype=float)
    if model_name == "croston_sba":
        value = croston_sba(values)
        return np.repeat(value, horizon), "Croston SBA"
    if model_name == "random_forest":
        if len(values) < 28:
            return seasonal_naive(values, horizon), "Seasonal Naive (short-history fallback)"
        try:
            return recursive_random_forest(ordered, horizon), "Random Forest"
        except ValueError:
            return seasonal_naive(values, horizon), "Seasonal Naive (training fallback)"
    if model_name == "seasonal_naive":
        return seasonal_naive(values, horizon), "Seasonal Naive"
    raise ValueError(f"Unknown model: {model_name}")
