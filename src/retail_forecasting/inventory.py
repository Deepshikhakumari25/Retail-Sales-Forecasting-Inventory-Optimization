"""Transparent safety-stock, EOQ, reorder-point and replenishment calculations."""

from __future__ import annotations

import math

from scipy.stats import norm


def _nonnegative(name: str, value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return numeric


def safety_stock(daily_demand_std: float, lead_time_days: float, service_level: float = 0.95) -> float:
    """Approximate safety stock as z * daily demand SD * sqrt(lead time)."""
    std = _nonnegative("daily_demand_std", daily_demand_std)
    lead = _nonnegative("lead_time_days", lead_time_days)
    if lead == 0 or not 0 < service_level < 1:
        raise ValueError("lead_time_days must be positive and service_level must be in (0, 1)")
    return float(norm.ppf(service_level) * std * math.sqrt(lead))


def reorder_point(daily_demand_mean: float, lead_time_days: float, buffer: float) -> float:
    """Expected lead-time demand plus safety stock."""
    mean = _nonnegative("daily_demand_mean", daily_demand_mean)
    lead = _nonnegative("lead_time_days", lead_time_days)
    stock = _nonnegative("buffer", buffer)
    if lead == 0:
        raise ValueError("lead_time_days must be positive")
    return mean * lead + stock


def economic_order_quantity(annual_demand: float, ordering_cost: float, annual_holding_cost_per_unit: float) -> float:
    """Wilson EOQ using annual demand and annual cost per order/unit."""
    demand = _nonnegative("annual_demand", annual_demand)
    order_cost = _nonnegative("ordering_cost", ordering_cost)
    holding = _nonnegative("annual_holding_cost_per_unit", annual_holding_cost_per_unit)
    if demand == 0:
        return 0.0
    if order_cost == 0 or holding == 0:
        raise ValueError("ordering_cost and annual_holding_cost_per_unit must be positive")
    return math.sqrt(2 * demand * order_cost / holding)


def replenishment_recommendation(
    *,
    daily_demand_mean: float,
    daily_demand_std: float,
    lead_time_days: float,
    current_stock: float,
    unit_cost: float | None = None,
    ordering_cost: float = 50.0,
    holding_rate: float = 0.22,
    incoming_orders: float = 0.0,
    backorders: float = 0.0,
    service_level: float = 0.95,
    review_period_days: int = 28,
) -> dict[str, float | int | str]:
    """Apply an order-up-to target over lead time plus review period.

    EOQ is a planning reference; the proposed quantity is not forced to EOQ.
    """
    mean = _nonnegative("daily_demand_mean", daily_demand_mean)
    std = _nonnegative("daily_demand_std", daily_demand_std)
    lead = _nonnegative("lead_time_days", lead_time_days)
    stock = _nonnegative("current_stock", current_stock)
    incoming = _nonnegative("incoming_orders", incoming_orders)
    backorders_value = _nonnegative("backorders", backorders)
    order_cost = _nonnegative("ordering_cost", ordering_cost)
    holding_rate_value = _nonnegative("holding_rate", holding_rate)
    if lead == 0 or review_period_days < 1 or not 0 < service_level < 1:
        raise ValueError("lead time/review period must be positive and service level in (0, 1)")
    if unit_cost is not None:
        unit_cost = _nonnegative("unit_cost", unit_cost)
    buffer = safety_stock(std, lead, service_level)
    expected = mean * lead
    point = expected + buffer
    position = stock + incoming - backorders_value
    annual_demand = mean * 365
    annual_holding = unit_cost * holding_rate_value if unit_cost is not None else 0.0
    eoq = economic_order_quantity(annual_demand, order_cost, annual_holding) if annual_demand > 0 and annual_holding > 0 else 0.0
    target = mean * (lead + review_period_days) + buffer
    quantity = max(0, int(math.ceil(target - position))) if position <= point else 0
    status = "ORDER" if quantity > 0 else "NO ORDER"
    return {
        "expected_lead_time_demand": expected,
        "safety_stock": buffer,
        "reorder_point": point,
        "eoq": eoq,
        "inventory_position": position,
        "order_up_to_target": target,
        "recommended_order_qty": quantity,
        "estimated_order_value": quantity * unit_cost if unit_cost is not None else float("nan"),
        "status": status,
        "explanation": (
            f"Position {position:.1f} is at/below reorder point {point:.1f}; "
            f"order to the {lead + review_period_days:.0f}-day target."
            if quantity else f"Position {position:.1f} is above reorder point {point:.1f}; no order."
        ),
    }
