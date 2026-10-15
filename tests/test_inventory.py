import math

import pytest

from retail_forecasting.inventory import economic_order_quantity, reorder_point, replenishment_recommendation, safety_stock


def test_safety_stock_reorder_point_and_eoq():
    assert safety_stock(2, 4, 0.95) == pytest.approx(6.58, abs=0.02)
    assert reorder_point(5, 4, 7) == 27
    assert economic_order_quantity(3650, 50, 2) == pytest.approx(math.sqrt(182500))


def test_replenishment_order_policy_and_inventory_position():
    no_order = replenishment_recommendation(
        daily_demand_mean=2, daily_demand_std=1, lead_time_days=5, current_stock=100,
    )
    assert no_order["recommended_order_qty"] == 0
    assert no_order["status"] == "NO ORDER"
    order = replenishment_recommendation(
        daily_demand_mean=2, daily_demand_std=1, lead_time_days=5, current_stock=0,
        incoming_orders=5, backorders=2, unit_cost=4,
    )
    assert order["recommended_order_qty"] > 0
    assert order["inventory_position"] == 3
    assert order["estimated_order_value"] == order["recommended_order_qty"] * 4
    assert order["eoq"] > 0


@pytest.mark.parametrize("kwargs", [
    {"daily_demand_mean": -1, "daily_demand_std": 0, "lead_time_days": 2, "current_stock": 1},
    {"daily_demand_mean": 1, "daily_demand_std": 0, "lead_time_days": 0, "current_stock": 1},
    {"daily_demand_mean": 1, "daily_demand_std": 0, "lead_time_days": 2, "current_stock": 1, "service_level": 1.0},
])
def test_invalid_inventory_values_are_rejected(kwargs):
    with pytest.raises(ValueError):
        replenishment_recommendation(**kwargs)


def test_eoq_rejects_zero_costs_when_demand_exists():
    with pytest.raises(ValueError):
        economic_order_quantity(100, 0, 2)
    with pytest.raises(ValueError):
        economic_order_quantity(100, 10, 0)


def test_zero_demand_and_variability_produce_zero_stock_buffer_and_order():
    assert safety_stock(0, 5) == 0
    recommendation = replenishment_recommendation(
        daily_demand_mean=0, daily_demand_std=0, lead_time_days=5, current_stock=0,
    )
    assert recommendation["eoq"] == 0
    assert recommendation["recommended_order_qty"] == 0
