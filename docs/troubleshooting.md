# Troubleshooting

## Python version and installation

Use Python 3.12 or 3.13; 3.13 was used for project validation. On Windows, check `py -0p`. Create a local environment with `py -3.13 -m venv .venv`, activate `.\.venv\Scripts\Activate.ps1`, then run `python -m pip install --upgrade pip` and `pip install -e ".[dev]"`. Python 3.14 is outside the declared project range.

If package installation fails, confirm internet access to the configured package index, inspect the full pip error, and retry using the project's `requirements.txt`. Do not install an incompatible package into the system Python.

On the validated Windows workspace, Application Control blocked a SciPy native extension installed inside `.venv`. Where compatible dependencies are already installed in the base Python, the working project environment can reuse them with `py -3.13 -m venv --system-site-packages .venv` followed by `.\.venv\Scripts\python.exe -m pip install -e . --no-deps`. This is environment-specific; otherwise use the isolated install with dependencies.

## Missing data or date parsing

Without a user upload, `python main.py` creates `data/sample/retail_sales.csv`; the dashboard uses that sample when present. Core CSV columns are `date`, `store_id`, `item_id`, `qty_sold`. Use dates that pandas can parse consistently. A malformed date, missing required fields or negative quantity produces an explicit error. Inspect `outputs/metrics/data_quality.json` for duplicates, missing values and stockout counts.

## Model training and forecasting

Random Forest needs at least 28 days of a selected series for a future forecast and uses Seasonal Naive as a documented fallback when history/training is too short. SBA handles sparse demand; an all-zero training series forecasts zero. If evaluation says there are too few dates, provide more history or reduce the evaluation test window in the call to `evaluate_models`.

The benchmark is a chronological rolling one-step holdout, not a random split. MASE can be undefined for flat training demand. Forecasts are expected units, not guaranteed sales.

## Inventory calculations

Lead time must be positive; demand, stock and cost values must be finite and non-negative. Service level must be strictly between 0 and 1. EOQ is zero when annual demand is zero and is not calculated with zero holding/order cost. Synthetic costs are illustrative; use vetted, consistent real costs before any real planning.

## Plotting and Streamlit

Pipeline chart generation selects Matplotlib's non-interactive `Agg` backend. If a chart fails, check that Matplotlib and Seaborn are installed and that `outputs/figures` is writable. Start the UI from project root with `streamlit run app\streamlit_app.py`. If Streamlit is not found, activate the project environment and run `python -m streamlit run app\streamlit_app.py`.

## Imports and tests

From project root, use `pip install -e ".[dev]"`, or use the documented direct command. For a direct source checkout the scripts add `src` to the import path; pytest does likewise when run from project root with editable installation. Run `python -m pytest -q`; read the failing assertion and traceback before changing expected behavior.
