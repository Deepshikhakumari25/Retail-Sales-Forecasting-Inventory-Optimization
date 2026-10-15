# Architecture and data flow

1. **Input** — `data/sample/retail_sales.csv` is generated deterministically if absent. A user file in `data/raw/retail_sales.csv` takes precedence in the dashboard.
2. **Validation** — `data_loader.py` parses CSV data; `data_validation.py` enforces required identifiers, non-negative sales/costs and parseable dates. It reports duplicate keys and missing values rather than silently dropping records.
3. **Features** — `features.py` sorts each store/product history, creates calendar fields, uses lagged demand, and shifts by one day before rolling statistics.
4. **Evaluation** — `evaluation.py` uses chronological train, validation and final test periods. The validation window selects among Seasonal Naive, Random Forest and SBA; the later test window is reported but does not select the model. Both windows use rolling one-step predictions, so later-day features may use sales observed earlier in that same window.
5. **Forecast** — `forecasting.py` exposes Seasonal Naive, recursive Random Forest and SBA methods for a selected series. Recursive prediction updates the lag history only with prior observations and predictions. Unseen store/product categories use an unknown code during pooled evaluation.
6. **Inventory** — `inventory.py` computes lead-time demand, safety stock, reorder point, EOQ reference and an order-up-to quantity.
7. **Outputs** — `main.py` writes forecasts, evaluation/quality CSV or JSON, inventory and purchase order CSVs, and charts under `outputs/`. The dashboard renders the same data and allows CSV download.

The Python package is under `src/retail_forecasting`; `main.py` and the Streamlit entry point add `src` to the import path for direct workspace execution. Install editable (`pip install -e ".[dev]"`) for normal package use.
