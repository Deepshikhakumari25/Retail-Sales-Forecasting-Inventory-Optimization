"""CSV loading and offline synthetic-data fallback."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data_validation import prepare_data
from .simulation import generate_synthetic_data


def load_retail_data(path: str | Path | None = None, *, fallback: bool = True) -> pd.DataFrame:
    """Load and validate CSV data; generate reproducible fallback if requested."""
    if path is None or not Path(path).exists():
        if not fallback:
            raise FileNotFoundError(f"Dataset not found: {path}")
        return prepare_data(generate_synthetic_data())
    return prepare_data(pd.read_csv(path))
