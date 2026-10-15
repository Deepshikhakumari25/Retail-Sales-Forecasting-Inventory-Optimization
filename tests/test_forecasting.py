import numpy as np
import pandas as pd
import pytest

from retail_forecasting.evaluation import _scores, evaluate_models
from retail_forecasting.forecasting import forecast_series, recursive_random_forest, seasonal_naive
from retail_forecasting.intermittent import croston_sba


def history(days=100):
    dates = pd.date_range("2024-01-01", periods=days)
    return pd.DataFrame({
        "date": dates, "store_id": "S1", "item_id": "P1",
        "qty_sold": [2 + (date.dayofweek == 5) * 3 for date in dates],
        "promo_flag": 0,
    })


def test_seasonal_naive_validates_and_repeats_cycle():
    np.testing.assert_array_equal(seasonal_naive([1, 2, 3, 4, 5, 6, 7], 9), [1, 2, 3, 4, 5, 6, 7, 1, 2])
    assert seasonal_naive([4, 6], 3).tolist() == [5, 5, 5]
    with pytest.raises(ValueError):
        seasonal_naive([], 3)


def test_croston_sba_sparse_and_zero_series():
    assert croston_sba([0, 0, 0]) == 0
    assert 0 < croston_sba([0, 4, 0, 0, 0, 8, 0]) < 4


def test_random_forest_forecast_and_chronological_evaluation():
    data = history(130)
    prediction, label = forecast_series(data, 14, "random_forest")
    assert len(prediction) == 14 and (prediction >= 0).all()
    assert label == "Random Forest"
    metrics, metadata = evaluate_models(data, test_days=20)
    assert set(metrics.model) == {"Seasonal Naive", "Random Forest", "Croston SBA"}
    assert metadata["train_end"] < metadata["test_start"]
    assert metadata["train_end"] < metadata["validation_start"] < metadata["test_start"]
    assert metadata["validation_end"] < metadata["test_start"]
    assert metrics.loc[metrics["selected_on_validation"], "model"].iloc[0] == metadata["selected_model"]
    assert metadata["selected_model"] in set(metrics.model)


def test_recursive_random_forest_rejects_short_training_series():
    with pytest.raises(ValueError):
        recursive_random_forest(history(10), 7)


def test_random_forest_uses_documented_short_history_fallback():
    predictions, label = forecast_series(history(20), 7, "random_forest")
    assert (predictions >= 0).all()
    assert label == "Seasonal Naive (short-history fallback)"


def test_forecast_requires_positive_horizon():
    with pytest.raises(ValueError, match="horizon"):
        forecast_series(history(), 0)


def test_mase_uses_valid_per_series_seasonal_scales():
    scores = _scores(
        np.array([2.0, 2.0, 4.0]),
        np.array([1.0, 0.0, 2.0]),
        np.array([1.0, 2.0, 0.0]),
    )
    assert scores["mase"] == pytest.approx(1.0)


def test_chronological_evaluation_handles_new_product_after_training():
    existing = history(160)
    new_product = existing.iloc[100:].copy()
    new_product["item_id"] = "P-NEW"
    new_product["qty_sold"] = 1.0
    panel = pd.concat([existing, new_product], ignore_index=True)
    metrics, metadata = evaluate_models(panel, test_days=20, validation_days=56)
    assert metadata["selected_model"] in set(metrics["model"])
    assert metrics["mae"].notna().all()
