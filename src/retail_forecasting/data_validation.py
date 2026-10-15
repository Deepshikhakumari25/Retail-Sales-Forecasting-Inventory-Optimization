"""Input checks and quality summaries for retail observations."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

REQUIRED_COLUMNS = ("date", "store_id", "item_id", "qty_sold")
NUMERIC_NONNEGATIVE = ("qty_sold", "price", "unit_cost", "ordering_cost", "lead_time_days", "stock_on_hand")


def validate_columns(data: pd.DataFrame) -> None:
    """Raise a readable error when core fields are absent."""
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """Parse dates and apply documented optional-field defaults."""
    validate_columns(data)
    if data.empty:
        raise ValueError("Dataset is empty")
    result = data.copy()
    result["date"] = pd.to_datetime(result["date"], format="mixed", errors="coerce")
    if result["date"].isna().any():
        count = int(result["date"].isna().sum())
        raise ValueError(f"{count} date value(s) could not be parsed")
    for column in NUMERIC_NONNEGATIVE:
        if column in result:
            original_missing = result[column].isna()
            result[column] = pd.to_numeric(result[column], errors="coerce")
            if column == "price":
                if (result[column].isna() & ~original_missing).any():
                    raise ValueError("price contains non-numeric values")
                if (result[column].dropna() < 0).any():
                    raise ValueError("price cannot contain negative values")
                continue
            if result[column].isna().any():
                raise ValueError(f"{column} contains missing or non-numeric values")
            if (result[column] < 0).any():
                raise ValueError(f"{column} cannot contain negative values")
    if "promo_flag" in result:
        if result["promo_flag"].isna().any():
            result["promo_flag"] = result["promo_flag"].fillna(0)
        if not result["promo_flag"].isin([0, 1, False, True]).all():
            raise ValueError("promo_flag must contain only 0/1 values")
        result["promo_flag"] = result["promo_flag"].astype(int)
    result["qty_sold"] = pd.to_numeric(result["qty_sold"], errors="coerce")
    if result["qty_sold"].isna().any():
        raise ValueError("qty_sold contains missing or non-numeric values")
    if (result["qty_sold"] < 0).any():
        raise ValueError("qty_sold cannot contain negative values")
    return result.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)


def data_quality_summary(data: pd.DataFrame) -> dict[str, object]:
    """Return counts of missing data, duplicates, stockouts and unusual values."""
    validate_columns(data)
    dates = pd.to_datetime(data["date"], errors="coerce")
    missing_series_dates = 0
    valid = data.loc[dates.notna(), ["store_id", "item_id"]].copy()
    valid["date"] = dates.loc[dates.notna()]
    for _, group in valid.groupby(["store_id", "item_id"], sort=False):
        observed_dates = group["date"].drop_duplicates()
        if not observed_dates.empty:
            expected_days = (observed_dates.max() - observed_dates.min()).days + 1
            missing_series_dates += max(0, expected_days - len(observed_dates))
    summary: dict[str, object] = {
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "missing_by_column": {str(k): int(v) for k, v in data.isna().sum().items()},
        "invalid_dates": int(dates.isna().sum()),
        "missing_store_item_dates": int(missing_series_dates),
        "duplicate_store_item_date": int(data.duplicated(["store_id", "item_id", "date"]).sum()),
        "negative_qty_sold": int((pd.to_numeric(data["qty_sold"], errors="coerce") < 0).sum()),
        "stockout_observations": int(data["stockout_flag"].fillna(0).astype(bool).sum()) if "stockout_flag" in data else None,
        "zero_sales_observations": int((pd.to_numeric(data["qty_sold"], errors="coerce") == 0).sum()),
        "unusual_qty_sold_over_99th_percentile": 0,
    }
    quantity = pd.to_numeric(data["qty_sold"], errors="coerce")
    if quantity.notna().any():
        summary["unusual_qty_sold_over_99th_percentile"] = int((quantity > quantity.quantile(0.99)).sum())
    return summary


def validate_inventory_inputs(values: Mapping[str, object]) -> None:
    """Raise for negative/non-finite optional inventory cost assumptions."""
    for key in ("unit_cost", "ordering_cost", "holding_rate", "lead_time_days"):
        if key in values and values[key] is not None:
            value = float(values[key])
            if not pd.notna(value) or value < 0:
                raise ValueError(f"{key} must be a finite non-negative value")
