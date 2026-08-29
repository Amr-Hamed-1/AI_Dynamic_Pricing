"""
Price optimiser for the Dynamic Pricing project.

Given a product, a competitor price and an optional cost, it:

1. Takes that product's *historical context* (its typical weight, volume,
   score, competitor prices, seasonality, ...) as the base feature vector.
2. Sweeps a range of candidate prices around the competitor price.
3. Predicts the expected quantity for each candidate price with the trained
   model, then computes revenue and profit.
4. Picks the price that maximises profit (or revenue when no cost is given).
5. Reports a *lift* versus the "do nothing" baseline (keeping the product's
   current price), which is the headline business KPI.

Run:
    python src/optimization.py
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import joblib
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

# Columns that should never leak into a prediction row
LEAK_COLS = [
    "qty", "revenue", "total_price", "freight_price",
    "unit_price", "comp_1", "comp_2", "comp_3",
    "price_diff_1", "price_diff_2", "price_diff_3",
]


class PriceOptimizer:
    def __init__(self, model_path: str, data_path: str):
        print(f"🔄 Loading model from: {model_path}")
        self.model = joblib.load(model_path)
        print(f"🔄 Loading data for context from: {data_path}")
        self.df = pd.read_csv(data_path)

        try:
            self.model_features = self.model.feature_names_in_
        except AttributeError:
            self.model_features = self.model.get_booster().feature_names

        self.products = sorted(
            self.df["product_id"].astype(str).unique().tolist()
        )

    # ------------------------------------------------------------------
    def _product_context(self, product_id: str, month: int = 6) -> pd.DataFrame:
        """Return a single context row for a product (its typical attributes)."""
        prod = self.df[self.df["product_id"].astype(str) == product_id]
        if prod.empty:
            raise ValueError(f"Unknown product: {product_id}")

        # Average this product's own rows to get a stable 'typical' context,
        # then drop anything we will overwrite per candidate price.
        base = prod.mean(numeric_only=True).to_frame().T
        drop = [c for c in LEAK_COLS if c in base.columns]
        context = base.drop(columns=drop)
        context["month"] = month
        return context

    # ------------------------------------------------------------------
    def get_optimal_price(self, competitor_price: float, product_cost: float = 0.0,
                          product_id: Optional[str] = None,
                          month: int = 6):
        if product_id is None:
            product_id = self.products[0]

        min_price = competitor_price * 0.7
        max_price = competitor_price * 1.3
        possible_prices = np.linspace(min_price, max_price, 60)

        base_data = self._product_context(product_id, month=month)
        # Keep the product's *current* price to use as the "do nothing" baseline
        current_price = float(base_data.get("lag_price", competitor_price).iloc[0]) \
            if "lag_price" in base_data.columns else float(competitor_price)

        results = []
        print(f"🧪 Testing {len(possible_prices)} different prices...")

        for price in possible_prices:
            row = base_data.copy()
            row["unit_price"] = price
            row["comp_1"] = competitor_price
            row["comp_2"] = competitor_price
            row["comp_3"] = competitor_price
            row["price_diff_1"] = price - competitor_price
            row["price_diff_2"] = price - competitor_price
            row["price_diff_3"] = price - competitor_price

            final_input = row.reindex(columns=self.model_features, fill_value=0)

            predicted_qty = self.model.predict(final_input)[0]
            predicted_qty = max(0, float(predicted_qty))

            revenue = predicted_qty * price
            profit = (price - product_cost) * predicted_qty if product_cost > 0 else revenue

            results.append({
                "price": price,
                "predicted_qty": predicted_qty,
                "revenue": revenue,
                "profit": profit,
            })

        results_df = pd.DataFrame(results)

        metric = "profit" if product_cost > 0 else "revenue"
        best_scenario = results_df.sort_values(by=metric, ascending=False).iloc[0].to_dict()

        # ---- Baseline lift --------------------------------------------------
        baseline = self._predict_at(base_data, current_price,
                                    competitor_price, product_cost)
        lift = self._compute_lift(baseline, best_scenario, metric)

        summary = {
            "product_id": product_id,
            "competitor_price": competitor_price,
            "product_cost": product_cost,
            "current_price": current_price,
            "best_price": best_scenario["price"],
            "best_qty": best_scenario["predicted_qty"],
            "best_revenue": best_scenario["revenue"],
            "best_profit": best_scenario["profit"] if product_cost > 0 else None,
            "baseline_qty": baseline["qty"],
            "baseline_revenue": baseline["revenue"],
            "baseline_profit": baseline["profit"] if product_cost > 0 else None,
            "lift": lift,
            "optimal_by": metric,
        }
        return summary, results_df

    # ------------------------------------------------------------------
    def _predict_at(self, context: pd.DataFrame, price: float,
                    competitor_price: float, product_cost: float) -> dict:
        row = context.copy()
        row["unit_price"] = price
        row["comp_1"] = competitor_price
        row["comp_2"] = competitor_price
        row["comp_3"] = competitor_price
        row["price_diff_1"] = price - competitor_price
        row["price_diff_2"] = price - competitor_price
        row["price_diff_3"] = price - competitor_price

        final_input = row.reindex(columns=self.model_features, fill_value=0)
        qty = max(0, float(self.model.predict(final_input)[0]))
        revenue = qty * price
        profit = (price - product_cost) * qty if product_cost > 0 else revenue
        return {"qty": qty, "revenue": revenue, "profit": profit}

    @staticmethod
    def _compute_lift(baseline: dict, best: dict, metric: str) -> dict:
        base_val = baseline[metric] if baseline[metric] else 1.0
        best_val = best[metric]
        return {
            f"{metric}_pct": ((best_val - baseline[metric]) / base_val) * 100,
            "qty_units": best["predicted_qty"] - baseline["qty"],
        }


# ---------------------------------------------------------------------------
# (Main Block)
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    model_path = os.path.join(PARENT_DIR, "models", "best_model.pkl")
    data_path = os.path.join(PARENT_DIR, "data", "processed",
                             "processed_retail_price.csv")

    if not (os.path.exists(model_path) and os.path.exists(data_path)):
        print("❌ Error: Model or Data file not found!")
        raise SystemExit(1)

    optimizer = PriceOptimizer(model_path, data_path)

    product_id = optimizer.products[0]
    competitor_price = float(
        optimizer.df[optimizer.df["product_id"].astype(str) == product_id]["comp_1"].median()
    )
    product_cost = 50.0

    print(f"\n--- 🎯 Optimization scenario ---")
    print(f"Product       : {product_id}")
    print(f"Competitor    : {competitor_price:.2f}$")
    print(f"Product cost  : {product_cost:.2f}$")

    summary, results_df = optimizer.get_optimal_price(
        competitor_price, product_cost, product_id=product_id
    )

    print(f"\n✅ Recommended Price : {summary['best_price']:.2f}$")
    print(f"📦 Expected Sales    : {summary['best_qty']:.2f} units")
    print(f"💰 Expected Revenue  : {summary['best_revenue']:.2f}$")
    print(f"💵 Expected Profit   : {summary['best_profit']:.2f}$")
    print(f"\n📊 Baseline (keep current price {summary['current_price']:.2f}$):")
    print(f"   qty={summary['baseline_qty']:.2f}, "
          f"revenue={summary['baseline_revenue']:.2f}$")
    lift_key = f"{summary['optimal_by']}_pct"
    print(f"🚀 {summary['optimal_by'].title()} lift vs baseline : "
          f"{summary['lift'][lift_key]:+.1f}%")
    print(f"   Quantity change          : {summary['lift']['qty_units']:+.1f} units")

    if summary["best_price"] < competitor_price:
        print("💡 Strategy: Undercut (أرخص من المنافس)")
    else:
        print("💡 Strategy: Premium (أغلى، الجودة تشفع لك)")
