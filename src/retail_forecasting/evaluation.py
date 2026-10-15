"""Three-way chronological evaluation with validation-only model selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .features import FEATURE_COLUMNS, add_demand_features
from .forecasting import seasonal_naive
from .intermittent import croston_sba


def _scores(actual: np.ndarray, predicted: np.ndarray, mase_scale: np.ndarray) -> dict[str, float]:
    errors = actual - predicted
    valid_scale = np.isfinite(mase_scale) & (mase_scale > 0)
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mase": float(np.mean(np.abs(errors[valid_scale]) / mase_scale[valid_scale])) if valid_scale.any() else float("nan"),
    }


def _rolling_intermittent_predictions(
    history: pd.DataFrame, period: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Predict one day ahead per series, updating only with earlier actuals."""
    seasonal = np.empty(len(period), dtype=float)
    croston = np.empty(len(period), dtype=float)
    histories = {
        key: group.sort_values("date")["qty_sold"].astype(float).tolist()
        for key, group in history.groupby(["store_id", "item_id"], sort=False)
    }
    fallback = float(history["qty_sold"].mean()) if not history.empty else 0.0
    for key, group in period.groupby(["store_id", "item_id"], sort=False):
        values = histories.get(key, [fallback] * 7).copy()
        for original_index, row in group.sort_values("date").iterrows():
            position = period.index.get_loc(original_index)
            seasonal[position] = seasonal_naive(values, 1)[0]
            croston[position] = croston_sba(values)
            values.append(float(row["qty_sold"]))
    return seasonal, croston


def _scales_by_series(data: pd.DataFrame) -> dict[tuple[object, object], float]:
    scales: dict[tuple[object, object], float] = {}
    for key, group in data.groupby(["store_id", "item_id"], sort=False):
        values = group.sort_values("date")["qty_sold"].to_numpy(dtype=float)
        scales[key] = float(np.mean(np.abs(values[7:] - values[:-7]))) if len(values) > 7 else float("nan")
    return scales


def _aligned_scales(period: pd.DataFrame, scales: dict[tuple[object, object], float]) -> np.ndarray:
    return np.asarray(
        [scales.get((row["store_id"], row["item_id"]), float("nan")) for _, row in period.iterrows()],
        dtype=float,
    )


def _fit_forest(featured: pd.DataFrame) -> RandomForestRegressor:
    usable = featured.dropna(subset=["lag_1", "lag_7", "lag_14"])
    if len(usable) < 10:
        raise ValueError("Insufficient training history for Random Forest")
    model = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(usable[list(FEATURE_COLUMNS)], usable["qty_sold"])
    return model


def evaluate_models(
    data: pd.DataFrame,
    test_days: int = 56,
    validation_days: int = 56,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Select on a validation window and report final metrics on later test dates.

    Predictions within validation/test are rolling one-step forecasts: actual
    demand observed earlier in the same window is available to later days.
    """
    if test_days < 1 or validation_days < 1:
        raise ValueError("test_days and validation_days must be positive")
    dates = pd.to_datetime(data["date"])
    test_start = dates.max() - pd.Timedelta(days=test_days - 1)
    validation_end = test_start - pd.Timedelta(days=1)
    validation_start = validation_end - pd.Timedelta(days=validation_days - 1)
    train = data.loc[dates < validation_start].copy()
    validation = data.loc[(dates >= validation_start) & (dates < test_start)].copy()
    test = data.loc[dates >= test_start].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Not enough history for chronological train, validation and test periods")

    store_codes = {value: index for index, value in enumerate(sorted(train["store_id"].unique()))}
    item_codes = {value: index for index, value in enumerate(sorted(train["item_id"].unique()))}
    featured = add_demand_features(data, store_codes=store_codes, item_codes=item_codes)
    feature_dates = pd.to_datetime(featured["date"])
    train_features = featured.loc[feature_dates < validation_start]
    validation_features = featured.loc[(feature_dates >= validation_start) & (feature_dates < test_start)]
    validation_forest = _fit_forest(train_features)
    validation_forest_prediction = np.maximum(
        0.0, validation_forest.predict(validation_features[list(FEATURE_COLUMNS)])
    )
    validation_seasonal, validation_croston = _rolling_intermittent_predictions(train, validation)
    validation_actual = validation["qty_sold"].to_numpy(dtype=float)
    validation_scale = _aligned_scales(validation, _scales_by_series(train))

    validation_predictions = {
        "Seasonal Naive": validation_seasonal,
        "Random Forest": validation_forest_prediction,
        "Croston SBA": validation_croston,
    }
    validation_scores = {
        name: _scores(validation_actual, prediction, validation_scale)
        for name, prediction in validation_predictions.items()
    }
    selected = min(validation_scores, key=lambda name: validation_scores[name]["mae"])

    development = data.loc[dates < test_start].copy()
    development_features = featured.loc[feature_dates < test_start]
    test_features = featured.loc[feature_dates >= test_start]
    test_forest = _fit_forest(development_features)
    test_forest_prediction = np.maximum(0.0, test_forest.predict(test_features[list(FEATURE_COLUMNS)]))
    test_seasonal, test_croston = _rolling_intermittent_predictions(development, test)
    test_predictions = {
        "Seasonal Naive": test_seasonal,
        "Random Forest": test_forest_prediction,
        "Croston SBA": test_croston,
    }
    test_actual = test["qty_sold"].to_numpy(dtype=float)
    test_scale = _aligned_scales(test, _scales_by_series(development))
    rows = []
    for name in validation_predictions:
        rows.append({
            "model": name,
            "validation_mae": validation_scores[name]["mae"],
            "validation_rmse": validation_scores[name]["rmse"],
            "validation_mase": validation_scores[name]["mase"],
            "mae": _scores(test_actual, test_predictions[name], test_scale)["mae"],
            "rmse": _scores(test_actual, test_predictions[name], test_scale)["rmse"],
            "mase": _scores(test_actual, test_predictions[name], test_scale)["mase"],
            "selected_on_validation": name == selected,
        })
    comparison = test[["date", "store_id", "item_id", "qty_sold"]].copy()
    comparison["seasonal_naive"] = test_seasonal
    comparison["random_forest"] = test_forest_prediction
    comparison["croston_sba"] = test_croston
    validation_comparison = validation[["date", "store_id", "item_id", "qty_sold"]].copy()
    validation_comparison["seasonal_naive"] = validation_seasonal
    validation_comparison["random_forest"] = validation_forest_prediction
    validation_comparison["croston_sba"] = validation_croston
    metadata = {
        "train_start": str(train["date"].min().date()),
        "train_end": str(train["date"].max().date()),
        "validation_start": str(validation["date"].min().date()),
        "validation_end": str(validation["date"].max().date()),
        "test_start": str(test["date"].min().date()),
        "test_end": str(test["date"].max().date()),
        "selected_model": selected,
        "selection_metric": "lowest aggregate rolling one-step validation MAE; final test was not used for selection",
        "comparison": comparison,
        "validation_comparison": validation_comparison,
        "forest": test_forest,
    }
    return pd.DataFrame(rows), metadata
