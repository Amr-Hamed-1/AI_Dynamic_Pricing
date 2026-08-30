"""
Optimizer Backtest for the Dynamic Pricing project.

What it does
------------
Proves that the PriceOptimizer actually adds value *on data it never saw*.

We take the temporal **test set** (the last 20% of rows by month/year that
`train.py` withheld from training) and, for each held-out product-row:

  1. Treat the row's own features as the decision context (its competitor
     prices, seasonality, product attributes, ...).
  2. Sweep candidate prices around the competitor price with the trained model.
  3. Compute profit if we "do nothing" (keep the row's actual price).
  4. Compute profit if we apply the optimizer's recommended price.
  5. Measure the **profit lift** vs. doing nothing.

Honest framing
--------------
We don't have real price A/B experiments, so the model acts as the *demand
oracle* (simulator). This is a **model-based policy backtest**: it answers
"given the demand model, does the optimizer's pricing policy earn more than
keeping current prices?" — a standard evaluation for pricing agents. It is
*not* a field A/B test, and the README says so.

Outputs
-------
- A printed summary (average % profit lift, % of rows improved, incremental $).
- `models/backtest_report.json` with per-product and aggregated results.

Run:
    python src/backtest.py
"""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
ROOT = PARENT_DIR

# Columns that leak the target or act as identifiers — never model features.
DROPPED = [
    "qty", "revenue", "total_price", "freight_price",
    "product_id", "month_year",
]
# Columns the optimizer overwrites per candidate price (unit price-driven).
OVERWRITTEN = [
    "unit_price",
    "comp_1", "comp_2", "comp_3",
    "price_diff_1", "price_diff_2", "price_diff_3",
]

TRAIN_RATIO = 0.8          # must match train.py
SWEEP_POINTS = 60          # candidate prices per row (as in optimization.py)
COST_RATIO = 0.4           # unit cost = 40% of the product's current price
SEED = 42


def _expected_profit(qty: float, price: float, cost: float) -> float:
    return max(0.0, (price - cost)) * qty


def _feature_vector(row: pd.Series, model_features: list) -> dict:
    """Build a model-ready feature dict from an actual held-out row.

    Removes leak/identity columns, then lets the caller overwrite the
    price-dependent columns per candidate.
    """
    vec = {c: row[c] for c in model_features if c in row.index}
    return vec


def load_split_data(data_path: str):
    data = pd.read_csv(data_path)
    data = data.sort_values(by=["month", "year"]).reset_index(drop=True)
    split_idx = int(len(data) * TRAIN_RATIO)
    test = data.iloc[split_idx:].copy()
    return test


def _sweep(model, model_features: list, row: pd.Series, comp_price: float, cost: float):
    """Sweep candidate prices and return scenarios + best-by-profit."""
    base = _feature_vector(row, model_features)
    min_price, max_price = comp_price * 0.7, comp_price * 1.3
    prices = np.linspace(min_price, max_price, SWEEP_POINTS)

    scenarios = []
    for price in prices:
        v = dict(base)
        v["unit_price"] = price
        v["comp_1"] = comp_price
        v["comp_2"] = comp_price
        v["comp_3"] = comp_price
        diff = price - comp_price
        v["price_diff_1"] = diff
        v["price_diff_2"] = diff
        v["price_diff_3"] = diff

        X = pd.DataFrame([v]).reindex(columns=model_features, fill_value=0)
        qty = max(0.0, float(model.predict(X)[0]))
        scenarios.append({
            "price": float(price),
            "qty": qty,
            "profit": _expected_profit(qty, price, cost),
        })

    best = max(scenarios, key=lambda s: s["profit"])
    return scenarios, best


def _baseline(model, model_features: list, row: pd.Series, comp_price: float,
              current_price: float, cost: float) -> dict:
    v = _feature_vector(row, model_features)
    v["unit_price"] = current_price
    v["comp_1"] = comp_price
    v["comp_2"] = comp_price
    v["comp_3"] = comp_price
    diff = current_price - comp_price
    v["price_diff_1"] = diff
    v["price_diff_2"] = diff
    v["price_diff_3"] = diff

    X = pd.DataFrame([v]).reindex(columns=model_features, fill_value=0)
    qty = max(0.0, float(model.predict(X)[0]))
    return {"price": current_price, "qty": qty,
            "profit": _expected_profit(qty, current_price, cost)}


def run_backtest(model_path: str, data_path: str,
                 n_products: int | None = None) -> dict:
    model = joblib.load(model_path)
    model_features = list(model.feature_names_in_)

    test = load_split_data(data_path)

    # One decision row per product (use each product's last held-out row so
    # every product appears once and no product dominates the average).
    test_sorted = test.sort_values(by=["month", "year"], ascending=True)
    last_rows = test_sorted.groupby("product_id", as_index=False).tail(1)

    if n_products is not None and n_products < len(last_rows):
        rng = np.random.default_rng(SEED)
        last_rows = last_rows.sample(n=n_products, random_state=SEED)

    per_product = []
    improved = 0
    total_baseline_profit = 0.0
    total_optimized_profit = 0.0

    for _, row in last_rows.iterrows():
        product_id = str(row["product_id"])
        comp_price = float(row["comp_1"])
        current_price = float(row["unit_price"])
        cost = COST_RATIO * current_price

        baseline = _baseline(model, model_features, row, comp_price,
                             current_price, cost)
        _, best = _sweep(model, model_features, row, comp_price, cost)

        lift_pct = ((best["profit"] - baseline["profit"]) / baseline["profit"]) * 100 \
            if baseline["profit"] > 0 else float("nan")
        is_improved = best["profit"] > baseline["profit"] + 1e-9
        if is_improved:
            improved += 1

        total_baseline_profit += baseline["profit"]
        total_optimized_profit += best["profit"]

        per_product.append({
            "product_id": product_id,
            "month": int(row["month"]), "year": int(row["year"]),
            "competitor_price": round(comp_price, 2),
            "current_price": round(current_price, 2),
            "baseline_profit": round(baseline["profit"], 2),
            "recommended_price": round(best["price"], 2),
            "recommended_profit": round(best["profit"], 2),
            "profit_lift_pct": round(lift_pct, 2) if not np.isnan(lift_pct) else None,
            "improved": is_improved,
        })

    n = len(per_product)
    avg_lift = np.mean([p["profit_lift_pct"] for p in per_product
                        if p["profit_lift_pct"] is not None])
    report = {
        "method": "model-based policy backtest (model acts as demand oracle \
on the held-out temporal test set; not a field A/B test)",
        "n_products_evaluated": n,
        "train_ratio": TRAIN_RATIO,
        "cost_ratio": COST_RATIO,
        "sweep_points": SWEEP_POINTS,
        "aggregate": {
            "avg_profit_lift_pct": round(float(avg_lift), 2),
            "pct_products_improved": round(improved / n * 100, 2),
            "total_baseline_profit": round(total_baseline_profit, 2),
            "total_optimized_profit": round(total_optimized_profit, 2),
            "total_incremental_profit": round(
                total_optimized_profit - total_baseline_profit, 2),
        },
        "per_product": per_product,
    }
    return report


def main() -> None:
    model_path = os.path.join(ROOT, "models", "best_model.pkl")
    data_path = os.path.join(ROOT, "data", "processed",
                             "processed_retail_price.csv")
    if not (os.path.exists(model_path) and os.path.exists(data_path)):
        print("❌ Model or data not found. Run generate_data.py + train.py first.")
        raise SystemExit(1)

    report = run_backtest(model_path, data_path)

    print("\n=== 📈 Optimizer Backtest (held-out temporal test set) ===")
    print(f"Products evaluated : {report['n_products_evaluated']}")
    agg = report["aggregate"]
    print(f"Avg profit lift    : {agg['avg_profit_lift_pct']:+.1f}%")
    print(f"Products improved  : {agg['pct_products_improved']:.1f}%")
    print("Model-based policy backtest: the demand model acts as the oracle on "
          "unseen rows (not a field A/B test).")

    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    out = os.path.join(ROOT, "models", "backtest_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"✅ Backtest report written to {out}")


if __name__ == "__main__":
    main()
