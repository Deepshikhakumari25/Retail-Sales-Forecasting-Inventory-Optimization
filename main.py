"""Run the reproducible retail data-to-recommendation pipeline."""

from __future__ import annotations

import json
import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from retail_forecasting.data_loader import load_retail_data
from retail_forecasting.data_validation import data_quality_summary
from retail_forecasting.evaluation import evaluate_models
from retail_forecasting.forecasting import forecast_series
from retail_forecasting.inventory import replenishment_recommendation
from retail_forecasting.reporting import export_purchase_order
from retail_forecasting.simulation import generate_synthetic_data


def _recommendations(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (store, item), group in data.groupby(["store_id", "item_id"]):
        recent = group.sort_values("date").tail(90)
        latest = group.sort_values("date").iloc[-1]
        values = recent["qty_sold"].astype(float)
        result = replenishment_recommendation(
            daily_demand_mean=float(values.mean()),
            daily_demand_std=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            lead_time_days=float(latest.get("lead_time_days", 7)),
            current_stock=float(latest.get("stock_on_hand", 0)),
            unit_cost=float(latest["unit_cost"]) if pd.notna(latest.get("unit_cost")) else None,
            ordering_cost=float(latest.get("ordering_cost", 50)),
            holding_rate=float(latest.get("holding_rate", 0.22)),
        )
        rows.append({
            "store_id": store, "item_id": item, "category": latest.get("category", "Unknown"),
            "current_stock": float(latest.get("stock_on_hand", 0)), "lead_time_days": latest.get("lead_time_days", 7),
            "daily_demand_mean": float(values.mean()), "daily_demand_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "unit_cost": latest.get("unit_cost", float("nan")), "data_source": latest.get("data_source", "user_csv"),
            **result,
        })
    return pd.DataFrame(rows)


def _create_figures(data: pd.DataFrame, comparison: pd.DataFrame, recommendations: pd.DataFrame) -> None:
    figures = ROOT / "outputs" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    daily = data.groupby("date", as_index=False)["qty_sold"].sum()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(daily["date"], daily["qty_sold"], color="#2563eb", linewidth=1)
    ax.set(title="Daily Units Sold Across Stores and Products", xlabel="Date", ylabel="Units sold")
    fig.tight_layout()
    fig.savefig(figures / "sales_trend.png", dpi=150)
    plt.close(fig)

    preview = data.head(6)[["date", "store_id", "item_id", "qty_sold", "stock_on_hand"]].copy()
    preview["date"] = pd.to_datetime(preview["date"]).dt.strftime("%Y-%m-%d")
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.axis("off")
    ax.table(cellText=preview.values, colLabels=preview.columns, loc="center", cellLoc="center")
    ax.set_title("Dataset Preview — Generated Sample Records")
    fig.tight_layout()
    fig.savefig(figures / "dataset_preview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    quality = data_quality_summary(data)
    quality_values = {
        "Missing cells": sum(quality["missing_by_column"].values()),
        "Duplicate keys": quality["duplicate_store_item_date"],
        "Invalid dates": quality["invalid_dates"],
        "Missing series dates": quality["missing_store_item_dates"],
        "Stockout rows": quality["stockout_observations"] or 0,
        "Zero-sales rows": quality["zero_sales_observations"],
    }
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(x=list(quality_values.values()), y=list(quality_values), color="#0f766e", ax=ax)
    ax.set(title="Data Quality Counts (zero/stockout counts are context, not errors)", xlabel="Rows / cells", ylabel="")
    fig.tight_layout()
    fig.savefig(figures / "data_quality_report.png", dpi=150)
    plt.close(fig)

    categories = data.groupby("category", as_index=False)["qty_sold"].sum().sort_values("qty_sold", ascending=False) if "category" in data else None
    if categories is not None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        sns.barplot(data=categories, x="qty_sold", y="category", color="#0f766e", ax=ax)
        ax.set(title="Total Units Sold by Category", xlabel="Units sold", ylabel="Category")
        fig.tight_layout()
        fig.savefig(figures / "category_analysis.png", dpi=150)
        plt.close(fig)

    monthly = data.assign(month=pd.to_datetime(data["date"]).dt.to_period("M").astype(str)).groupby("month")["qty_sold"].sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    monthly.plot(ax=ax, color="#d97706", marker="o")
    ax.set(title="Monthly Units Sold", xlabel="Month", ylabel="Units sold")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(figures / "monthly_demand.png", dpi=150)
    plt.close(fig)

    weekly = data.assign(weekday=pd.to_datetime(data["date"]).dt.day_name()).groupby("weekday")["qty_sold"].mean()
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    fig, ax = plt.subplots(figsize=(9, 4))
    weekly.reindex(weekdays).plot(kind="bar", ax=ax, color="#7c3aed")
    ax.set(title="Average Demand by Weekday", xlabel="Day of week", ylabel="Average units per observation")
    fig.tight_layout()
    fig.savefig(figures / "weekly_seasonality.png", dpi=150)
    plt.close(fig)

    selected = comparison.head(120)
    if len(selected):
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(pd.to_datetime(selected["date"]), selected["qty_sold"], label="Actual", color="#111827")
        ax.plot(pd.to_datetime(selected["date"]), selected["random_forest"], label="Random Forest (one-step)", alpha=0.8)
        ax.plot(pd.to_datetime(selected["date"]), selected["seasonal_naive"], label="Seasonal Naive", alpha=0.8)
        ax.set(title="Chronological Holdout: Actual vs Predictions (sample rows)", xlabel="Date", ylabel="Units")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / "forecast_comparison.png", dpi=150)
        plt.close(fig)

    model_scores = pd.read_csv(ROOT / "outputs" / "metrics" / "model_metrics.csv")
    fig, ax = plt.subplots(figsize=(9, 4))
    positions = np.arange(len(model_scores))
    width = 0.36
    ax.bar(positions - width / 2, model_scores["validation_mae"], width, label="Validation (select)", color="#0f766e")
    ax.bar(positions + width / 2, model_scores["mae"], width, label="Final test", color="#2563eb")
    ax.set_xticks(positions, model_scores["model"])
    ax.set(title="Validation Model Selection and Untouched Test MAE", xlabel="Model", ylabel="MAE (units)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "model_evaluation.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = recommendations["status"].map({"ORDER": "#dc2626", "NO ORDER": "#059669"})
    ax.bar(recommendations["item_id"] + " / " + recommendations["store_id"], recommendations["recommended_order_qty"], color=colors)
    ax.set(title="Simulated Replenishment Recommendations", xlabel="Product / store", ylabel="Recommended units")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(figures / "reorder_alerts.png", dpi=150)
    plt.close(fig)

    table_data = recommendations[["store_id", "item_id", "current_stock", "reorder_point", "recommended_order_qty", "status"]].head(12)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis("off")
    ax.table(cellText=table_data.round(1).values, colLabels=table_data.columns, loc="center", cellLoc="center")
    ax.set_title("Inventory Recommendations — Simulated Assumptions")
    fig.tight_layout()
    fig.savefig(figures / "inventory_recommendations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    purchase = recommendations.loc[recommendations["recommended_order_qty"] > 0, ["store_id", "item_id", "recommended_order_qty", "unit_cost", "estimated_order_value"]].head(12)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axis("off")
    if len(purchase):
        ax.table(cellText=purchase.round(2).values, colLabels=purchase.columns, loc="center", cellLoc="center")
    else:
        ax.text(0.5, 0.5, "No order lines for this dataset", ha="center", va="center")
    ax.set_title("Purchase Order Export Preview — Simulated Costs")
    fig.tight_layout()
    fig.savefig(figures / "purchase_order_export.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")
    overview = (
        f"Retail Sales Forecasting & Inventory Optimization\n\n"
        f"Observations: {len(data):,}     Stores: {data['store_id'].nunique()}     Products: {data['item_id'].nunique()}\n"
        f"Units sold: {data['qty_sold'].sum():,.0f}     Validation-selected model: {model_scores.loc[model_scores.selected_on_validation, 'model'].iloc[0]}\n"
        f"Products/store with order recommendation: {(recommendations['recommended_order_qty'] > 0).sum()}\n\n"
        "Synthetic-data educational simulation — not measured business savings"
    )
    ax.text(0.5, 0.5, overview, ha="center", va="center", fontsize=13, wrap=True)
    fig.tight_layout()
    fig.savefig(figures / "project_overview.png", dpi=150)
    plt.close(fig)


def run_pipeline() -> dict[str, object]:
    """Generate all reproducible project outputs and return a compact summary."""
    directories = [
        ROOT / "data" / "sample", ROOT / "outputs" / "forecasts", ROOT / "outputs" / "inventory",
        ROOT / "outputs" / "metrics", ROOT / "outputs" / "figures", ROOT / "images",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    sample_path = ROOT / "data" / "sample" / "retail_sales.csv"
    if not sample_path.exists():
        generate_synthetic_data().to_csv(sample_path, index=False)
    raw_path = ROOT / "data" / "raw" / "retail_sales.csv"
    active_path = raw_path if raw_path.exists() else sample_path
    data = load_retail_data(active_path, fallback=False)
    data.head(100).to_csv(ROOT / "outputs" / "metrics" / "dataset_preview.csv", index=False)
    quality = data_quality_summary(data)
    (ROOT / "outputs" / "metrics" / "data_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")

    scores, evaluation = evaluate_models(data)
    scores.to_csv(ROOT / "outputs" / "metrics" / "model_metrics.csv", index=False)
    comparison = evaluation.pop("comparison")
    validation_comparison = evaluation.pop("validation_comparison")
    evaluation.pop("forest")
    (ROOT / "outputs" / "metrics" / "evaluation_metadata.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    comparison.to_csv(ROOT / "outputs" / "metrics" / "holdout_predictions.csv", index=False)
    validation_comparison.to_csv(ROOT / "outputs" / "metrics" / "validation_predictions.csv", index=False)

    store, item = data[["store_id", "item_id"]].drop_duplicates().iloc[0]
    series = data.loc[(data["store_id"] == store) & (data["item_id"] == item)].sort_values("date")
    model_key = {"Seasonal Naive": "seasonal_naive", "Random Forest": "random_forest", "Croston SBA": "croston_sba"}[evaluation["selected_model"]]
    predictions, model_name = forecast_series(series, 28, model_key)
    forecast_dates = pd.date_range(series["date"].max() + pd.Timedelta(days=1), periods=28, freq="D")
    forecast = pd.DataFrame({
        "forecast_date": forecast_dates, "store_id": store, "item_id": item,
        "predicted_demand": predictions, "model_name": model_name,
        "rounding_note": "predictions shown as fractional expected units; procurement quantity is rounded up",
    })
    forecast.to_csv(ROOT / "outputs" / "forecasts" / "demand_forecast.csv", index=False)

    recommendations = _recommendations(data)
    recommendations.to_csv(ROOT / "outputs" / "inventory" / "recommendations.csv", index=False)
    export_purchase_order(recommendations, ROOT / "outputs" / "inventory" / "purchase_order.csv")
    recommendations.to_csv(ROOT / "outputs" / "metrics" / "inventory_recommendations.csv", index=False)
    _create_figures(data, comparison, recommendations)
    # Portfolio assets are genuine data-derived charts, not GUI screenshots.
    for source, target in (
        ("project_overview.png", "01_project_overview.png"),
        ("dataset_preview.png", "02_dataset_preview.png"),
        ("data_quality_report.png", "03_data_quality_report.png"),
        ("sales_trend.png", "04_sales_trend.png"),
        ("category_analysis.png", "05_category_analysis.png"),
        ("forecast_comparison.png", "06_forecast_comparison.png"),
        ("model_evaluation.png", "07_model_evaluation.png"),
        ("inventory_recommendations.png", "08_inventory_recommendations.png"),
        ("reorder_alerts.png", "09_reorder_alerts.png"),
        ("purchase_order_export.png", "11_purchase_order_export.png"),
    ):
        shutil.copy2(ROOT / "outputs" / "figures" / source, ROOT / "images" / target)

    eda_summary = {
        "total_units_sold": int(data["qty_sold"].sum()),
        "observations": int(len(data)),
        "stores": int(data["store_id"].nunique()),
        "products": int(data["item_id"].nunique()),
        "category_units": data.groupby("category")["qty_sold"].sum().astype(int).to_dict() if "category" in data else {},
        "top_products_by_units": data.groupby("item_id")["qty_sold"].sum().sort_values(ascending=False).head(5).astype(int).to_dict(),
        "intermittent_products_zero_share": data.groupby("item_id")["qty_sold"].apply(lambda values: float((values == 0).mean())).sort_values(ascending=False).to_dict(),
        "promo_mean_units": data.groupby("promo_flag")["qty_sold"].mean().to_dict() if "promo_flag" in data else {},
    }
    (ROOT / "outputs" / "metrics" / "eda_summary.json").write_text(json.dumps(eda_summary, indent=2), encoding="utf-8")
    return {
        "rows": len(data),
        "stores": int(data["store_id"].nunique()),
        "products": int(data["item_id"].nunique()),
        "selected_model": evaluation["selected_model"],
        "metrics": scores.to_dict(orient="records"),
        "forecast_rows": len(forecast),
        "orders": int((recommendations["recommended_order_qty"] > 0).sum()),
        "output_directory": str(ROOT / "outputs"),
    }


if __name__ == "__main__":
    print(json.dumps(run_pipeline(), indent=2))
