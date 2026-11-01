"""
Automated test suite for the Dynamic Pricing & Demand Forecasting pipeline.

Covers:
1. Chronological train/test split integrity (no future leakage across years)
2. Per-product lag_price isolation in synthetic generator
3. Absence of post-decision target-leakage columns (`customers`, `revenue`, `total_price`)
4. Economic monotonicity of predicted demand (`qty` never rises when price rises)
5. Vectorized PriceOptimizer profit maximization & lift consistency
6. FastAPI endpoint handlers & error responses

Run:
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import pandas as pd
from fastapi import HTTPException

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import api
from src.backtest import run_backtest
from src.generate_data import generate_frame, true_expected_demand
from src.optimization import PriceOptimizer
from src.preprocessing import DataPreprocessor, temporal_train_test_split


class TestDynamicPricingPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = os.path.join(ROOT, "models", "best_model.pkl")
        cls.data_path = os.path.join(
            ROOT, "data", "processed", "processed_retail_price.csv"
        )
        assert os.path.exists(cls.model_path), "Run src/train.py before running tests"
        assert os.path.exists(cls.data_path), "Run src/generate_data.py before running tests"
        cls.optimizer = PriceOptimizer(cls.model_path, cls.data_path)

    def test_chronological_split_has_no_future_leakage(self):
        df = pd.read_csv(self.data_path)
        train_df, test_df = temporal_train_test_split(df, train_ratio=0.8)

        max_train_period = int((train_df["year"] * 100 + train_df["month"]).max())
        min_test_period = int((test_df["year"] * 100 + test_df["month"]).min())

        self.assertLess(
            max_train_period,
            min_test_period,
            f"Temporal leak: max train period ({max_train_period}) >= min test period ({min_test_period})",
        )

    def test_lag_price_isolated_per_product(self):
        raw = generate_frame(n_products=10, months=6, seed=123)
        for _, group in raw.groupby("product_id"):
            ordered = group.sort_values(["year", "month"]).reset_index(drop=True)
            # Every month t >= 1 must match month t-1's unit_price of the SAME product
            np.testing.assert_allclose(
                ordered["lag_price"].iloc[1:].to_numpy(),
                ordered["unit_price"].iloc[:-1].to_numpy(),
            )

    def test_category_preserved_and_no_leakage_features_in_model(self):
        forbidden = {
            "qty",
            "revenue",
            "total_price",
            "freight_price",
            "customers",
            "product_id",
            "month_year",
        }
        model_feats = set(self.optimizer.model_features)
        leaked = forbidden.intersection(model_feats)
        self.assertEqual(leaked, set(), f"Forbidden leakage columns found in model: {leaked}")

        category_dummies = [c for c in model_feats if c.startswith("cat_")]
        self.assertGreater(len(category_dummies), 0, "Expected one-hot category features (`cat_*`)")
        self.assertIn("price_ratio", model_feats)

    def test_monotonic_demand_elasticity_curve(self):
        """When XGBoost is selected with monotonic constraints, raising price never raises qty."""
        inner = getattr(self.optimizer.model, "regressor_", self.optimizer.model)
        if type(inner).__name__ != "XGBRegressor":
            self.skipTest("Monotonic constraint assertion applies to XGBRegressor")

        for pid in self.optimizer.products[:5]:
            _, curve = self.optimizer.get_optimal_price(
                competitor_price=120.0, product_cost=45.0, product_id=pid, month=11
            )
            diffs = np.diff(curve["predicted_qty"].to_numpy())
            self.assertTrue(
                np.all(diffs <= 1e-4),
                f"Non-monotonic demand jump detected for {pid}: max diff={diffs.max()}",
            )

    def test_optimizer_maximizes_profit_above_cost(self):
        summary, curve = self.optimizer.get_optimal_price(
            competitor_price=150.0,
            product_cost=60.0,
            product_id=self.optimizer.products[0],
            month=6,
        )
        self.assertEqual(summary["optimal_by"], "profit")
        self.assertGreater(summary["best_price"], 60.0)
        self.assertAlmostEqual(
            summary["best_profit"], float(curve["profit"].max()), places=5
        )
        # If current_price sits inside the swept range, best_profit must >= baseline_profit
        if curve["price"].min() <= summary["current_price"] <= curve["price"].max():
            self.assertGreaterEqual(
                summary["best_profit"] + 1e-3, summary["baseline_profit"]
            )

    def test_preprocessing_compatibility(self):
        raw = generate_frame(n_products=10, months=4, seed=7)
        proc = DataPreprocessor(df=raw).run()
        self.assertIn("price_ratio", proc.columns)
        self.assertNotIn("month_year", proc.columns)
        self.assertTrue(any(c.startswith("cat_") for c in proc.columns))

    def test_generator_matches_ground_truth_demand(self):
        """qty in the raw data must be the true demand function plus small noise."""
        raw = generate_frame(n_products=20, months=6, seed=11)
        comp_avg = raw[["comp_1", "comp_2", "comp_3"]].mean(axis=1)
        expected = np.array([
            true_expected_demand(r.product_category_name, r.volume, r.product_score,
                                 r.month, r.unit_price, ca)
            for r, ca in zip(raw.itertuples(), comp_avg)
        ])
        ratio = raw["qty"].to_numpy() / expected
        self.assertAlmostEqual(float(np.median(ratio)), 1.0, delta=0.1)

    def test_backtest_reports_ground_truth_within_oracle_ceiling(self):
        report = run_backtest(self.model_path, self.data_path, n_products=40)
        gt = report["aggregate"]["ground_truth"]
        self.assertIn("model_estimated", report["aggregate"])
        # No policy can beat the oracle under true demand
        self.assertLessEqual(gt["portfolio_profit_lift_pct"], gt["oracle_ceiling_lift_pct"] + 1e-6)
        self.assertGreaterEqual(gt["oracle_ceiling_lift_pct"], 0.0)

    def test_fastapi_handlers(self):
        h = api.health()
        self.assertEqual(h["status"], "ok")
        self.assertTrue(h["optimizer_loaded"])

        prods = api.list_products()
        self.assertGreater(len(prods["products"]), 0)

        info = api.model_info()
        self.assertEqual(info["n_features"], len(self.optimizer.model_features))
        self.assertIn("XGBoost", info["metrics"].get("metrics", {}))

        req = api.OptimizeRequest(
            product_id=prods["products"][0],
            competitor_price=130.0,
            product_cost=50.0,
            month=12,
        )
        res = api.optimize(req)
        self.assertGreater(res["best_price"], 50.0)
        self.assertEqual(res["optimal_by"], "profit")

        with self.assertRaises(HTTPException) as ctx:
            api.optimize(
                api.OptimizeRequest(
                    product_id="non_existent_sku",
                    competitor_price=100.0,
                    product_cost=30.0,
                )
            )
        self.assertEqual(ctx.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
