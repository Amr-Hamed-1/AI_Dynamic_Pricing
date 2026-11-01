"""
Price optimiser for the Dynamic Pricing project.

Given a product, a competitor price, and an optional unit cost (COGS), it:
1. Builds that product's historical feature context from training history only
   (no test-set leakage).
2. Evaluates 60 candidate prices around the competitor price in a single
   vectorized batch prediction call.
3. Picks the price that maximises profit (or revenue when cost == 0).
4. Computes lift vs. keeping the product's current baseline price.

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
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.preprocessing import temporal_train_test_split

# Columns that either leak post-decision outcomes or get overwritten per candidate price
LEAK_COLS = [
    "qty",
    "revenue",
    "total_price",
    "freight_price",
    "customers",
    "unit_price",
    "comp_1",
    "comp_2",
    "comp_3",
    "price_diff_1",
    "price_diff_2",
    "price_diff_3",
    "price_ratio",
]


class PriceOptimizer:
    def __init__(self, model_path: str, data_path: str):
        self.model = joblib.load(model_path)
        self.df = pd.read_csv(data_path)

        # Use historical training split for product context to avoid test-set leakage
        train_df, _ = temporal_train_test_split(self.df, train_ratio=0.8)
        self.context_df = train_df if not train_df.empty else self.df

        try:
            self.model_features = list(self.model.feature_names_in_)
        except AttributeError:
            self.model_features = list(self.model.get_booster().feature_names)

        self.products = sorted(self.df["product_id"].astype(str).unique().tolist())

    # ------------------------------------------------------------------
    def _product_context(self, product_id: str, month: int = 6) -> pd.DataFrame:
        """Return a single historical context row for a product."""
        prod = self.context_df[self.context_df["product_id"].astype(str) == product_id]
        if prod.empty:
            prod = self.df[self.df["product_id"].astype(str) == product_id]
        if prod.empty:
            raise ValueError(f"Unknown product: {product_id}")

        # Latest historical price serves as the "do nothing" current price baseline
        ordered_prod = prod.sort_values(by=["year", "month"])
        latest_price = float(ordered_prod["unit_price"].iloc[-1])

        base = prod.mean(numeric_only=True).to_frame().T
        # Preserve binary one-hot category indicators cleanly
        for col in base.columns:
            if col.startswith("cat_"):
                base[col] = base[col].round().astype(int)

        drop = [c for c in LEAK_COLS if c in base.columns]
        context = base.drop(columns=drop)
        context["month"] = int(month)
        context["_current_price"] = latest_price
        return context

    # ------------------------------------------------------------------
    def _build_batch(
        self, context: pd.DataFrame, prices: np.ndarray, competitor_price: float
    ) -> pd.DataFrame:
        batch = pd.concat([context] * len(prices), ignore_index=True)
        batch["unit_price"] = prices
        batch["comp_1"] = competitor_price
        batch["comp_2"] = competitor_price
        batch["comp_3"] = competitor_price
        diffs = prices - competitor_price
        batch["price_diff_1"] = diffs
        batch["price_diff_2"] = diffs
        batch["price_diff_3"] = diffs
        batch["price_ratio"] = prices / max(competitor_price, 1e-3)
        return batch.reindex(columns=self.model_features, fill_value=0)

    # ------------------------------------------------------------------
    def get_optimal_price(
        self,
        competitor_price: float,
        product_cost: float = 0.0,
        product_id: Optional[str] = None,
        month: int = 6,
        n_points: int = 60,
    ):
        if product_id is None:
            product_id = self.products[0]

        # Ensure candidate sweep stays strictly above product cost when feasible
        min_price = max(competitor_price * 0.7, product_cost * 1.02 if product_cost > 0 else 1.0)
        max_price = max(competitor_price * 1.3, min_price * 1.15)
        possible_prices = np.linspace(min_price, max_price, n_points)

        base_data = self._product_context(product_id, month=month)
        current_price = float(base_data["_current_price"].iloc[0])

        # Vectorized batch inference across all candidate prices in one call
        X_batch = self._build_batch(base_data, possible_prices, competitor_price)
        predicted_qtys = np.maximum(0.0, np.asarray(self.model.predict(X_batch), dtype=float))

        revenues = predicted_qtys * possible_prices
        profits = (
            (possible_prices - product_cost) * predicted_qtys
            if product_cost > 0
            else revenues
        )

        results_df = pd.DataFrame({
            "price": possible_prices,
            "predicted_qty": predicted_qtys,
            "revenue": revenues,
            "profit": profits,
        })

        metric = "profit" if product_cost > 0 else "revenue"
        best_idx = int(results_df[metric].idxmax())
        best_scenario = results_df.iloc[best_idx].to_dict()

        baseline = self._predict_at(base_data, current_price, competitor_price, product_cost)
        lift = self._compute_lift(baseline, best_scenario, metric)

        summary = {
            "product_id": product_id,
            "competitor_price": float(competitor_price),
            "product_cost": float(product_cost),
            "current_price": current_price,
            "best_price": float(best_scenario["price"]),
            "best_qty": float(best_scenario["predicted_qty"]),
            "best_revenue": float(best_scenario["revenue"]),
            "best_profit": float(best_scenario["profit"]) if product_cost > 0 else None,
            "baseline_qty": baseline["qty"],
            "baseline_revenue": baseline["revenue"],
            "baseline_profit": baseline["profit"] if product_cost > 0 else None,
            "lift": lift,
            "optimal_by": metric,
        }
        return summary, results_df

    # ------------------------------------------------------------------
    def _predict_at(
        self,
        context: pd.DataFrame,
        price: float,
        competitor_price: float,
        product_cost: float,
    ) -> dict:
        X_single = self._build_batch(context, np.array([price], dtype=float), competitor_price)
        qty = max(0.0, float(self.model.predict(X_single)[0]))
        revenue = qty * price
        profit = (price - product_cost) * qty if product_cost > 0 else revenue
        return {"qty": qty, "revenue": revenue, "profit": profit}

    @staticmethod
    def _compute_lift(baseline: dict, best: dict, metric: str) -> dict:
        base_val = baseline[metric]
        best_val = best[metric]
        pct = ((best_val - base_val) / abs(base_val)) * 100.0 if abs(base_val) > 1e-6 else 0.0
        return {
            f"{metric}_pct": float(pct),
            "qty_units": float(best["predicted_qty"] - baseline["qty"]),
        }


if __name__ == "__main__":
    model_path = os.path.join(PARENT_DIR, "models", "best_model.pkl")
    data_path = os.path.join(PARENT_DIR, "data", "processed", "processed_retail_price.csv")

    if not (os.path.exists(model_path) and os.path.exists(data_path)):
        print("❌ Error: Model or Data file not found!")
        raise SystemExit(1)

    optimizer = PriceOptimizer(model_path, data_path)
    product_id = optimizer.products[0]
    competitor_price = float(
        optimizer.df[optimizer.df["product_id"].astype(str) == product_id]["comp_1"].median()
    )
    product_cost = 50.0

    print("\n--- 🎯 Optimization scenario ---")
    print(f"Product       : {product_id}")
    print(f"Competitor    : {competitor_price:.2f}$")
    print(f"Product cost  : {product_cost:.2f}$")

    summary, _ = optimizer.get_optimal_price(
        competitor_price, product_cost, product_id=product_id
    )

    print(f"\n✅ Recommended Price : {summary['best_price']:.2f}$")
    print(f"📦 Expected Sales    : {summary['best_qty']:.2f} units")
    print(f"💰 Expected Revenue  : {summary['best_revenue']:.2f}$")
    print(f"💵 Expected Profit   : {summary['best_profit']:.2f}$")
    print(f"\n📊 Baseline (keep current price {summary['current_price']:.2f}$):")
    print(
        f"   qty={summary['baseline_qty']:.2f}, "
        f"revenue={summary['baseline_revenue']:.2f}$, "
        f"profit={summary['baseline_profit']:.2f}$"
    )
    lift_key = f"{summary['optimal_by']}_pct"
    print(
        f"🚀 {summary['optimal_by'].title()} lift vs baseline : "
        f"{summary['lift'][lift_key]:+.1f}%"
    )
    print(f"   Quantity change          : {summary['lift']['qty_units']:+.1f} units")
