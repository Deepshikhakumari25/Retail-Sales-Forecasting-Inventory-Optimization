import pandas as pd

from retail_forecasting.features import add_demand_features


def test_lags_and_rolling_features_use_only_prior_observations():
    data = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=35),
        "store_id": "S1", "item_id": "P1", "qty_sold": range(1, 36),
    })
    result = add_demand_features(data)
    row = result.iloc[28]
    assert row["lag_1"] == 28
    assert row["lag_7"] == 22
    assert row["rolling_mean_7"] == 25
    assert row["rolling_mean_28"] == 14.5
    assert result.iloc[0]["rolling_mean_7"] != result.iloc[0]["qty_sold"]


def test_series_are_sorted_independently():
    data = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-02", "2025-01-01", "2025-01-01", "2025-01-02"]),
        "store_id": ["S1", "S1", "S2", "S2"], "item_id": "P1", "qty_sold": [2, 1, 10, 20],
    })
    result = add_demand_features(data)
    assert result.loc[result.store_id == "S1", "lag_1"].iloc[1] == 1
    assert result.loc[result.store_id == "S2", "lag_1"].iloc[1] == 10


def test_unseen_store_and_product_categories_get_unknown_codes():
    data = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=2),
        "store_id": ["S1", "S2"], "item_id": ["P1", "P2"], "qty_sold": [1, 2],
    })
    result = add_demand_features(data, store_codes={"S1": 0}, item_codes={"P1": 0})
    assert result.loc[1, "store_code"] == -1
    assert result.loc[1, "item_code"] == -1
