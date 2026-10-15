import pandas as pd
import pytest

from retail_forecasting.data_validation import data_quality_summary, prepare_data, validate_columns
from retail_forecasting.data_loader import load_retail_data


def frame():
    return pd.DataFrame({
        "date": ["2025-01-01", "2025-01-02"],
        "store_id": ["S1", "S1"], "item_id": ["P1", "P1"],
        "qty_sold": [0, 2], "promo_flag": [None, 1],
    })


def test_missing_required_columns_are_reported():
    with pytest.raises(ValueError, match="item_id"):
        validate_columns(pd.DataFrame({"date": [], "store_id": [], "qty_sold": []}))


def test_dates_promo_and_duplicate_summary():
    data = prepare_data(pd.concat([frame(), frame().iloc[[0]]], ignore_index=True))
    assert pd.api.types.is_datetime64_any_dtype(data["date"])
    assert data["promo_flag"].iloc[0] == 0
    assert data_quality_summary(data)["duplicate_store_item_date"] == 1


@pytest.mark.parametrize("column,value", [("qty_sold", -1), ("unit_cost", -3), ("lead_time_days", -1)])
def test_negative_values_are_rejected(column, value):
    data = frame().drop(columns="promo_flag")
    data[column] = value
    with pytest.raises(ValueError, match="negative"):
        prepare_data(data)


def test_invalid_date_and_missing_file_behavior(tmp_path):
    data = frame()
    data.loc[0, "date"] = "not-a-date"
    with pytest.raises(ValueError, match="date"):
        prepare_data(data)
    with pytest.raises(FileNotFoundError):
        load_retail_data(tmp_path / "absent.csv", fallback=False)


def test_missing_price_is_reported_but_not_silently_filled():
    data = frame()
    data["price"] = [None, 2.5]
    result = prepare_data(data)
    assert pd.isna(result["price"].iloc[0])
    assert data_quality_summary(result)["missing_by_column"]["price"] == 1


def test_csv_loader_reads_and_parses_existing_file(tmp_path):
    path = tmp_path / "retail.csv"
    frame().drop(columns="promo_flag").to_csv(path, index=False)
    loaded = load_retail_data(path, fallback=False)
    assert len(loaded) == 2
    assert pd.api.types.is_datetime64_any_dtype(loaded["date"])


def test_missing_calendar_dates_are_counted_per_series():
    data = pd.DataFrame({
        "date": ["2025-01-01", "2025-01-03"],
        "store_id": ["S1", "S1"], "item_id": ["P1", "P1"], "qty_sold": [1, 2],
    })
    assert data_quality_summary(data)["missing_store_item_dates"] == 1


def test_empty_dataset_is_rejected_clearly():
    empty = pd.DataFrame(columns=["date", "store_id", "item_id", "qty_sold"])
    with pytest.raises(ValueError, match="empty"):
        prepare_data(empty)
