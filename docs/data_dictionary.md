# Data dictionary

| Field | Meaning | Notes |
|---|---|---|
| `date` | Observation date | Daily date, parsed to pandas datetime |
| `store_id` | Store identifier | Synthetic identifiers such as S01 |
| `item_id` | Product identifier | Synthetic identifiers such as P001 |
| `category` | Product category | Optional for uploaded data |
| `qty_sold` | Observed units sold | Non-negative; stockout-censored on flagged days |
| `price` | Simulated unit selling price | Synthetic cost units; no currency is specified |
| `promo_flag` | Promotion indicator | 0/1; missing optional flags default to 0 |
| `stock_on_hand` | Simulated stock following daily sales/replenishment | Snapshot convention in synthetic data |
| `stockout_flag` | Latent demand exceeded available inventory | A flagged record may hide unmet demand |
| `lead_time_days` | Supplier lead-time assumption | Positive days for inventory planning |
| `unit_cost` | Simulated purchase cost per unit | Used for holding cost and order value; no currency is specified |
| `ordering_cost` | Simulated fixed cost per order | EOQ input; no currency is specified |
| `holding_rate` | Annual holding-cost rate | Holding cost per unit/year = unit_cost × holding_rate |
| `data_source` | Data provenance label | `synthetic_simulation` for generated records |

The unique expected grain is one `store_id` + `item_id` + `date` row. Core user CSV fields are date, store_id, item_id and qty_sold; all other columns are optional. Missing or malformed date and demand values are errors. Prices are not imputed; negative numeric costs or quantities are rejected.
