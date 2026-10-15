"""Small output helpers shared by the command-line pipeline and dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_purchase_order(recommendations: pd.DataFrame, path: str | Path) -> Path:
    """Write actionable order lines, retaining costs and simulated data labels."""
    required = {"store_id", "item_id", "recommended_order_qty"}
    missing = required - set(recommendations.columns)
    if missing:
        raise ValueError(f"Purchase order data is missing columns: {', '.join(sorted(missing))}")
    output = recommendations.loc[recommendations["recommended_order_qty"] > 0].copy()
    columns = [
        name for name in ("store_id", "item_id", "recommended_order_qty", "unit_cost", "estimated_order_value", "status", "data_source")
        if name in output.columns
    ]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output[columns].to_csv(destination, index=False)
    return destination
