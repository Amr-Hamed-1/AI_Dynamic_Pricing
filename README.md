# 🤖 Dynamic Pricing & Demand Forecasting

An **end-to-end** system that automatically finds the optimal product price based on competitor pricing, using **demand forecasting** with machine learning, then a decision engine that picks the price maximizing **profit** (or revenue) and measures **lift** vs. keeping the current price.

---

## 📌 Business Value

In e-commerce, price isn't static — the "right" price depends on:
- Competitor prices
- Expected demand at each price point (price elasticity)

This project solves it in two stages:
1. **Model**: Predicts units sold (`qty`) for a given price in a given market context.
2. **Optimizer**: Sweeps candidate prices around the competitor price, predicts demand for each, computes `revenue = qty × price` and `profit = (price − cost) × qty`, picks the best, and compares it to a **baseline** (keeping the current price) to show **Lift**.

---

## 🗂️ Project Structure

```
Dynamic_Pricing_Project/
├── app.py                  # Streamlit dashboard (product picker + simulation)
├── requirements.txt
├── README.md
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── data/
│   ├── raw/                # retail_price.csv (raw, git-ignored)
│   └── processed/          # processed_retail_price.csv (ready for training, git-ignored)
├── src/
│   ├── generate_data.py    # Synthetic data generator with explicit price elasticity
│   ├── preprocessing.py    # Cleaning + feature engineering
│   ├── train.py            # Training + evaluation + model selection + feature importance
│   ├── optimization.py     # PriceOptimizer (price engine + lift calculation)
│   ├── backtest.py         # Optimizer backtest on the held-out test set
│   └── data_loader.py
├── models/
│   ├── best_model.pkl      # Best model (XGBoost)
│   ├── training_report.json # Metrics + feature importance
│   └── backtest_report.json # Optimizer backtest summary + per-product results
└── notebooks/
    └── Cleaning&Eda.ipynb
```

> `data/` and `models/` are git-ignored. They are regenerated locally by running `generate_data.py` then `train.py` (or at Docker build time).

---

## 🚀 Quick Start (Local)

```bash
# 1. Install requirements
pip install -r requirements.txt

# 2. Generate synthetic data (creates data/processed)
python src/generate_data.py

# 3. Train & evaluate (creates models/best_model.pkl)
python src/train.py

# 4. Run optimizer CLI (shows lift)
python src/optimization.py

# 5. Backtest the optimizer on the held-out test set (writes models/backtest_report.json)
python src/backtest.py

# 6. (Optional) Launch interactive dashboard
streamlit run app.py
```

### 📊 Experiment Tracking with MLflow

Every `src/train.py` run logs its metrics, hyperparameters and artifacts to a
local **MLflow** experiment (`dynamic-pricing`) stored in `sqlite:///mlflow.db`:

- Metrics for both models (RandomForest & XGBoost MAE / RMSE / R²)
- Split params (temporal, train/test sizes) and the best model name
- The saved model + training report as run artifacts

View the runs in the web UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# open http://localhost:5000
```

> MLflow is **optional**: if it isn't installed, `train.py` still works and just
> skips logging. Override the store with the `MLFLOW_TRACKING_URI` env var.

---

## 🐳 Run with Docker (Self-contained)

The image **generates data + trains the model at build time**, so it works out of the box without any local data.

```bash
# Build & run
docker compose up --build -d

# Open http://localhost:8501
```

Stop:
```bash
docker compose down
```

---

## 🧠 Methodology

- **Split**: **Temporal split** (first 80% months = train, last 20% = test) — mimics real usage (predict future demand) instead of random split.
- **Models**: `RandomForestRegressor` and `XGBoost` with **GridSearchCV** (5-fold) for hyperparameter tuning.
- **Metric**: RMSE (penalizes large errors) + MAE + R².
- **Key features**: `volume`, `customers`, and `price_diff_*` (difference between our price and competitors') are top drivers — exactly what dynamic pricing is about.

### 📊 Results (on synthetic data, 4000 rows)

| Model           | MAE   | RMSE  | R²   |
|-----------------|-------|-------|------|
| RandomForest    | 4.46  | 6.13  | 0.59 |
| **XGBoost**     | **3.73** | **5.24** | **0.70** |

Best model: **XGBoost** (temporal split, R² = 0.70).

### 📈 Optimizer Backtest (held-out test set)

To prove the optimizer actually adds value, `src/backtest.py` evaluates it on
the **held-out temporal test set** (the 20% of rows the model never trained
on). For each product it compares the optimizer's recommended price against
"doing nothing" (keeping the current price):

| Metric                    | Value  |
|---------------------------|--------|
| Products evaluated        | 200    |
| Average profit lift       | **+9.2%** |
| Products with profit lift | **97%**  |

> This is a **model-based policy backtest**: the demand model acts as the
> oracle on unseen rows. It is **not** a field A/B test (we don't have real
> counterfactual experiments), but it demonstrates the pricing policy earns
> more than keeping prices static — an honest, practical evaluation.

Full per-product results are in `models/backtest_report.json`.

### 🧪 What the model can't prove & how we handle it in production

The backtest proves the pricing **policy** beats static pricing *under the
model's demand forecasts* — it does **not** prove real-world outcomes. The
model can't know factors beyond its columns (promotions, ad spend, competitor
quality, buyer psychology), so "raise the price and sales hold" is an
*assumption of the model*, not a market fact.

There is no shortcut around that — the only way to validate real revenue is a
**field A/B test**:

1. Split products into a **treatment** group (apply the optimizer's price) and a
   **control** group (keep current price).
2. Measure *actual* profit over a defined window (e.g. 30 days).
3. If the treatment shows a statistically significant profit lift, roll it out
   to all products; otherwise, re-train / dampen the optimizer.

So the honest positioning is: **backtest de-risks the model, A/B test validates
it in the real world.** Because the backtest is run on rows the model never saw
and the caveat is documented, it's a realistic evaluation — not a hand-waved
claim.

---

## 💡 Example Output

```
Product       : prod_0000
Competitor    : $145.94
Product cost  : $50.00

✅ Recommended Price : $118.48
📦 Expected Sales    : 26.43 units
💵 Expected Profit   : $1,810.13

📊 Baseline (keep current price $131.67):
   qty=20.57, revenue=$2,708.77
🚀 Profit lift vs baseline : +7.7%
   Quantity change         : +5.9 units
💡 Strategy: Undercut (cheaper than competitor)
```

Meaning: dropping the price from $131.67 to $118.48 (slightly below competitor) increases total profit by **+7.7%**.

---

## 🖥️ Dashboard (Streamlit)

- Pick a **product**
- Set **competitor price** & **product cost**
- See: recommended price, expected units, expected profit, baseline comparison
- **Revenue/Profit curve** vs. price
- Strategy explanation (Undercut / Premium)

---

## 🧾 Reports

On every training run, `train.py` writes `models/training_report.json` with:
- Metrics for each model (MAE, RMSE, R²)
- Top 20 feature importances

`src/backtest.py` writes `models/backtest_report.json` with:
- Aggregate profit-lift KPIs (`avg_profit_lift_pct`, `pct_products_improved`, `total_incremental_profit`)
- Per-product recommended price vs. current price and realized lift

---

## ⚠️ Honest Notes

- Data is **synthetic** — generated to have a clear price-elasticity signal. In production, replace with real data of the same schema.
- 4000 rows is enough for demo but not for production.
- Target `qty` is continuous → **Regressors** used (an earlier version mistakenly used `RandomForestClassifier` and was fixed).
- Temporal split is harder than random (R² drops from ~0.88 to 0.70) but is the correct evaluation for forecasting.

---

Built with Python, scikit-learn, XGBoost, Streamlit, Plotly, pandas & MLflow.