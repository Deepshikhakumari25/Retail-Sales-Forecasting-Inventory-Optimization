"""Leakage-aware calendar, lag and rolling feature engineering."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

FEATURE_COLUMNS = (
    "day_of_week", "day_of_month", "week_of_year", "month", "is_weekend",
    "lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_mean_14",
    "rolling_mean_28", "rolling_std_7", "promo_flag", "store_code", "item_code",
)


def add_demand_features(
    data: pd.DataFrame,
    *,
    store_codes: Mapping[object, int] | None = None,
    item_codes: Mapping[object, int] | None = None,
) -> pd.DataFrame:
    """Create per-store/item lag features; rolling windows exclude current day."""
    required = {"date", "store_id", "item_id", "qty_sold"}
    if not required.issubset(data.columns):
        raise ValueError(f"Feature input requires columns: {', '.join(sorted(required))}")
    result = data.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    result = result.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    date = result["date"]
    result["day_of_week"] = date.dt.dayofweek
    result["day_of_month"] = date.dt.day
    result["week_of_year"] = date.dt.isocalendar().week.astype(int)
    result["month"] = date.dt.month
    result["is_weekend"] = date.dt.dayofweek.isin([5, 6]).astype(int)
    grouped = result.groupby(["store_id", "item_id"], sort=False)["qty_sold"]
    for lag in (1, 7, 14):
        result[f"lag_{lag}"] = grouped.shift(lag)
    prior = grouped.shift(1)
    for window in (7, 14, 28):
        result[f"rolling_mean_{window}"] = prior.groupby(
            [result["store_id"], result["item_id"]], sort=False
        ).transform(lambda values: values.rolling(window, min_periods=1).mean())
    result["rolling_std_7"] = prior.groupby(
        [result["store_id"], result["item_id"]], sort=False
    ).transform(lambda values: values.rolling(7, min_periods=2).std()).fillna(0.0)
    result["promo_flag"] = result["promo_flag"].fillna(0).astype(int) if "promo_flag" in result else 0
    result["store_code"] = (
        result["store_id"].map(store_codes).fillna(-1).astype(int)
        if store_codes is not None else pd.Categorical(result["store_id"]).codes
    )
    result["item_code"] = (
        result["item_id"].map(item_codes).fillna(-1).astype(int)
        if item_codes is not None else pd.Categorical(result["item_id"]).codes
    )
    return result
