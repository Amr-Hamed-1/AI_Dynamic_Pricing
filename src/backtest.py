"""
Optimizer Backtest for the Dynamic Pricing project.

Evaluates the pricing policy on the chronological held-out test set
(via `temporal_train_test_split`, matching `train.py` exactly). For each SKU's
last held-out month it:

1. Sweeps 60 candidate prices around the competitor average in one batched
   model call and picks the profit-maximizing price.
2. Scores the decision TWO ways:
   - **Model-estimated lift**: the demand model scores both the recommended and
     the current price. This is circular (the same model chose the price), so
     it is optimistic by construction — the "optimizer's curse".
   - **Ground-truth lift**: because the data is synthetic, we know the true
     demand function (`generate_data.true_expected_demand`). We score the same
     recommended price against reality. This is the honest number.
3. Also reports the oracle ceiling (best achievable profit under true demand)
   and how much of that achievable uplift the optimizer actually captured.

Cost assumption: unit cost = COST_RATIO x competitor average price (a market
level cost), so the current price is NOT optimal by construction and there is
real room for a pricing policy to add (or destroy) value.

Run:
    python src/backtest.py
"""
from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CURRENT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.generate_data import true_expected_demand
from src.preprocessing import temporal_train_test_split

TRAIN_RATIO = 0.8
SWEEP_POINTS = 60
COST_RATIO = 0.4  # unit cost = 40% of competitor average price
ORACLE_POINTS = 600
SEED = 42


def _profit(qty, price, cost):
    return np.maximum(0.0, np.asarray(price) - cost) * np.asarray(qty)


def _row_category(row: pd.Series) -> str | None:
    cat_cols = [c for c in row.index if c.startswith("cat_")]
    if not cat_cols:
        return None
    best = max(cat_cols, key=lambda c: row[c])
    return best[len("cat_"):] if row[best] > 0 else None


def _build_candidates_df(
    row: pd.Series, model_features: list[str], prices: np.ndarray
) -> pd.DataFrame:
    base_dict = {c: row[c] for c in model_features if c in row.index}
    batch = pd.DataFrame([base_dict] * len(prices))

    comp_1, comp_2, comp_3 = float(row["comp_1"]), float(row["comp_2"]), float(row["comp_3"])
    comp_avg = max((comp_1 + comp_2 + comp_3) / 3.0, 1e-3)

    batch["unit_price"] = prices
    batch["comp_1"], batch["comp_2"], batch["comp_3"] = comp_1, comp_2, comp_3
    batch["price_diff_1"] = prices - comp_1
    batch["price_diff_2"] = prices - comp_2
    batch["price_diff_3"] = prices - comp_3
    batch["price_ratio"] = prices / comp_avg
    return batch.reindex(columns=model_features, fill_value=0)


def _sweep_bounds(comp_avg: float, cost: float) -> tuple[float, float]:
    lo = max(comp_avg * 0.7, cost * 1.02)
    hi = max(comp_avg * 1.3, lo * 1.15)
    return lo, hi


def _evaluate_row(model, model_features: list[str], row: pd.Series, cost: float):
    comp_avg = float((row["comp_1"] + row["comp_2"] + row["comp_3"]) / 3.0)
    current_price = float(row["unit_price"])
    lo, hi = _sweep_bounds(comp_avg, cost)
    candidates = np.linspace(lo, hi, SWEEP_POINTS)

    # Baseline + all candidates in ONE batched predict call
    all_prices = np.concatenate(([current_price], candidates))
    qtys = np.maximum(0.0, np.asarray(
        model.predict(_build_candidates_df(row, model_features, all_prices)), dtype=float
    ))
    profits = _profit(qtys, all_prices, cost)
    best_idx = int(np.argmax(profits[1:]))
    return {
        "current_price": current_price,
        "recommended_price": float(candidates[best_idx]),
        "model_baseline_profit": float(profits[0]),
        "model_recommended_profit": float(profits[1:][best_idx]),
    }


def _ground_truth(row: pd.Series, category: str, cost: float, current: float, recommended: float):
    comp_avg = float((row["comp_1"] + row["comp_2"] + row["comp_3"]) / 3.0)

    def true_profit(prices):
        q = true_expected_demand(category, row["volume"], row["product_score"],
                                 int(row["month"]), prices, comp_avg)
        return _profit(q, prices, cost)

    lo, hi = _sweep_bounds(comp_avg, cost)
    oracle = float(true_profit(np.linspace(lo, hi, ORACLE_POINTS)).max())
    return {
        "true_baseline_profit": float(true_profit(current)),
        "true_recommended_profit": float(true_profit(recommended)),
        "true_oracle_profit": oracle,
    }


def _pct(new: float, old: float) -> float:
    return (new - old) / old * 100.0 if old > 0 else 0.0


def run_backtest(model_path: str, data_path: str, n_products: int | None = None) -> dict:
    model = joblib.load(model_path)
    try:
        model_features = list(model.feature_names_in_)
    except AttributeError:
        model_features = list(model.get_booster().feature_names)

    data = pd.read_csv(data_path)
    _, test = temporal_train_test_split(data, train_ratio=TRAIN_RATIO)
    last_rows = (test.sort_values(by=["year", "month"])
                     .groupby("product_id", as_index=False).tail(1))
    if n_products is not None and n_products < len(last_rows):
        last_rows = last_rows.sample(n=n_products, random_state=SEED)

    per_product = []
    for _, row in last_rows.iterrows():
        comp_avg = float((row["comp_1"] + row["comp_2"] + row["comp_3"]) / 3.0)
        cost = COST_RATIO * comp_avg
        res = _evaluate_row(model, model_features, row, cost)

        category = _row_category(row)
        if category is not None:
            res.update(_ground_truth(row, category, cost,
                                     res["current_price"], res["recommended_price"]))

        res.update({
            "product_id": str(row["product_id"]),
            "year": int(row["year"]), "month": int(row["month"]),
            "competitor_avg_price": comp_avg, "unit_cost": cost,
        })
        per_product.append({k: (round(v, 2) if isinstance(v, float) else v)
                            for k, v in res.items()})

    df = pd.DataFrame(per_product)
    agg = {
        "model_estimated": {
            "portfolio_profit_lift_pct": round(_pct(df.model_recommended_profit.sum(),
                                                    df.model_baseline_profit.sum()), 2),
            "pct_products_improved": round(float(
                (df.model_recommended_profit > df.model_baseline_profit).mean() * 100), 2),
        },
    }
    has_truth = "true_baseline_profit" in df.columns
    if has_truth:
        base, rec, orc = (df.true_baseline_profit.sum(), df.true_recommended_profit.sum(),
                          df.true_oracle_profit.sum())
        agg["ground_truth"] = {
            "portfolio_profit_lift_pct": round(_pct(rec, base), 2),
            "pct_products_improved": round(float(
                (df.true_recommended_profit > df.true_baseline_profit).mean() * 100), 2),
            "oracle_ceiling_lift_pct": round(_pct(orc, base), 2),
            "share_of_achievable_uplift_captured_pct": round(
                (rec - base) / (orc - base) * 100.0, 1) if orc > base else None,
            "total_baseline_profit": round(base, 2),
            "total_recommended_profit": round(rec, 2),
            "total_incremental_profit": round(rec - base, 2),
        }

    return {
        "method": (
            "Out-of-time policy backtest on the chronological held-out test set. "
            "'model_estimated' scores decisions with the same model that made them "
            "(circular, optimistic). 'ground_truth' scores them with the known "
            "synthetic demand function (honest)."
        ),
        "n_products_evaluated": int(len(df)),
        "train_ratio": TRAIN_RATIO,
        "cost_assumption": f"unit_cost = {COST_RATIO} x competitor average price",
        "sweep_points": SWEEP_POINTS,
        "aggregate": agg,
        "per_product": per_product,
    }


def main() -> None:
    model_path = os.path.join(ROOT, "models", "best_model.pkl")
    data_path = os.path.join(ROOT, "data", "processed", "processed_retail_price.csv")
    if not (os.path.exists(model_path) and os.path.exists(data_path)):
        print("❌ Model or data not found. Run generate_data.py + train.py first.")
        raise SystemExit(1)

    report = run_backtest(model_path, data_path)
    agg = report["aggregate"]

    print("\n=== 📈 Optimizer Backtest (chronological held-out test set) ===")
    print(f"SKUs evaluated              : {report['n_products_evaluated']}")
    m = agg["model_estimated"]
    print(f"[model-estimated] lift      : {m['portfolio_profit_lift_pct']:+.1f}%  "
          f"(improved {m['pct_products_improved']:.1f}%)  <- circular, optimistic")
    if "ground_truth" in agg:
        g = agg["ground_truth"]
        print(f"[ground truth]    lift      : {g['portfolio_profit_lift_pct']:+.1f}%  "
              f"(improved {g['pct_products_improved']:.1f}%)")
        print(f"[ground truth]    ceiling   : {g['oracle_ceiling_lift_pct']:+.1f}%  "
              f"(captured {g['share_of_achievable_uplift_captured_pct']}% of achievable uplift)")

    out = os.path.join(ROOT, "models", "backtest_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"✅ Backtest report written to {out}")


if __name__ == "__main__":
    main()
