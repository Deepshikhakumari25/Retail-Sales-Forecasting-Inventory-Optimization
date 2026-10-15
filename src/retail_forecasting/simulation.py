"""Reproducible synthetic retail sales and inventory data."""

from __future__ import annotations

import numpy as np
import pandas as pd


PRODUCTS = (
    ("P001", "Beverages", 24.0, 11.0, 4, 0.035),
    ("P002", "Snacks", 18.0, 8.0, 6, 0.025),
    ("P003", "Household", 7.0, 3.2, 9, 0.015),
    ("P004", "Personal Care", 5.0, 2.1, 12, 0.010),
    ("P005", "Bakery", 13.0, 5.0, 3, 0.18),
    ("P006", "Seasonal", 3.4, 1.2, 14, 0.34),
    ("P007", "Dairy", 20.0, 9.0, 5, 0.04),
    ("P008", "Specialty", 2.2, 0.9, 10, 0.55),
)


def generate_synthetic_data(
    start_date: str = "2024-01-01",
    periods: int = 730,
    stores: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate deterministic daily store-product sales with simulated stock.

    Stockout-day sales are observed sales, not uncensored demand. The simulation
    records those days explicitly; downstream models do not impute lost sales.
    """
    if periods < 1 or stores < 1:
        raise ValueError("periods and stores must be positive integers")
    dates = pd.date_range(start=start_date, periods=periods, freq="D")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for store_number in range(1, stores + 1):
        store_id = f"S{store_number:02d}"
        store_factor = float(rng.uniform(0.82, 1.22))
        for product_index, product in enumerate(PRODUCTS):
            item_id, category, base, unit_cost, lead, zero_probability = product
            product_factor = float(rng.uniform(0.88, 1.14))
            inventory = max(18, int(base * lead * 1.8))
            unit_price = round(unit_cost * float(rng.uniform(1.55, 2.0)), 2)
            ordering_cost = 45.0 + (product_index % 3) * 15.0
            holding_rate = 0.22
            for index, date in enumerate(dates):
                promo = int(rng.random() < 0.08)
                weekday_factor = (0.78, 0.86, 0.94, 1.00, 1.12, 1.32, 1.24)[date.weekday()]
                trend = 1.0 + 0.00025 * index
                annual = 1.0 + 0.13 * np.sin(2 * np.pi * index / 365.25 + product_index / 2)
                poisson_mean = max(0.05, base * store_factor * product_factor * weekday_factor * trend * annual * (1.45 if promo else 1.0))
                latent = int(rng.poisson(poisson_mean))
                if rng.random() < zero_probability:
                    latent = 0
                sold = min(latent, inventory)
                stockout = int(latent > inventory)
                inventory -= sold
                # A simple periodic simulated replenishment keeps stock observable.
                if date.weekday() == 0 and inventory < base * lead * 1.2:
                    inventory += max(int(base * lead * 2.2), 1)
                rows.append(
                    {
                        "date": date,
                        "store_id": store_id,
                        "item_id": item_id,
                        "category": category,
                        "qty_sold": sold,
                        "price": unit_price,
                        "promo_flag": promo,
                        "stock_on_hand": inventory,
                        "stockout_flag": stockout,
                        "lead_time_days": lead,
                        "unit_cost": unit_cost,
                        "ordering_cost": ordering_cost,
                        "holding_rate": holding_rate,
                        "data_source": "synthetic_simulation",
                    }
                )
    return pd.DataFrame(rows)
