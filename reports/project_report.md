# Project report

## Scope

This educational project demonstrates daily retail demand forecasting followed by transparent replenishment planning. It uses a reproducible synthetic dataset; results are technical experiments, not evidence of real business savings.

## Method

Data are validated for required columns, date parsing, negative quantities/costs, duplicate store/product/day observations, missingness and stockout flags. Calendar, lagged demand and shifted rolling features feed a Random Forest model. Seasonal Naive is a weekly baseline and SBA is an intermittent-demand reference. Models are compared chronologically on a final holdout using MAE, RMSE and training-scaled MASE where defined.

## Inventory policy

Recent observed demand and variability estimate expected lead-time demand and safety stock. The reorder point is their sum. When inventory position is at/below the point, order-up-to quantity targets lead-time demand plus a 28-day review period and safety stock. EOQ is separately displayed as a cost-based reference. Recommendations and costs from the included generator are simulated.

## Reproducibility and results

Run `python main.py` to regenerate sample data and output tables/charts. The verified seeded run produced 17,520 observations. A validation window selected the model; the later 56-day test window remained untouched for final comparison:

| Model | Validation MAE | Test MAE | Test RMSE | Test Panel MASE |
|---|---:|---:|---:|---:|
| Seasonal Naive | 5.695 | 6.007 | 9.062 | 1.109 |
| Random Forest (selected on validation) | 4.151 | 4.380 | 6.654 | 0.823 |
| Croston SBA | 5.176 | 5.381 | 7.704 | 0.963 |

The `outputs/metrics/model_metrics.csv`, `evaluation_metadata.json` and `data_quality.json` files contain run details. These scores measure only this synthetic experiment; no future performance or business impact is claimed.

## Limitations

Stockout demand is censored, future promotions are not known, and the holdout is one cutoff with rolling one-step evaluation. Inventory formulas assume fixed lead time and independent daily variability. Validate all assumptions and data before any real-world use.
