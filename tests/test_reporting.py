import pandas as pd
import pytest

from retail_forecasting.reporting import export_purchase_order


def test_purchase_order_export_filters_no_order_rows(tmp_path):
    frame = pd.DataFrame({
        "store_id": ["S1", "S1"], "item_id": ["P1", "P2"],
        "recommended_order_qty": [4, 0], "unit_cost": [2.0, 3.0],
        "estimated_order_value": [8.0, 0.0],
    })
    destination = export_purchase_order(frame, tmp_path / "orders.csv")
    result = pd.read_csv(destination)
    assert len(result) == 1
    assert result.loc[0, "item_id"] == "P1"


def test_purchase_order_requires_minimum_fields(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        export_purchase_order(pd.DataFrame({"item_id": ["P1"]}), tmp_path / "bad.csv")
