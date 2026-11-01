"""
Preprocessing & feature engineering pipeline for Dynamic Pricing.

- Cleans missing competitor prices
- Preserves product category information via one-hot encoding (instead of dropping it)
- Engineers competitor price gaps & relative price ratio features
- Provides a shared chronological train/test split utility (`temporal_train_test_split`)
  used by both `train.py` and `backtest.py` to prevent temporal data leakage.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import load_data


def temporal_train_test_split(
    df: pd.DataFrame, train_ratio: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split panel data strictly chronologically by (year, month).

    Sorting by `year * 100 + month` guarantees:
    1. No future year leaks into training (fixes the `sort_values(['month', 'year'])` bug).
    2. All products within the boundary month stay together on the same side of the split.
    """
    ordered = df.sort_values(by=["year", "month", "product_id"]).reset_index(drop=True)
    period_key = ordered["year"].astype(int) * 100 + ordered["month"].astype(int)
    unique_periods = np.sort(period_key.unique())

    if len(unique_periods) < 2:
        split_idx = int(len(ordered) * train_ratio)
        return ordered.iloc[:split_idx].copy(), ordered.iloc[split_idx:].copy()

    cutoff_idx = max(1, min(len(unique_periods) - 1, int(len(unique_periods) * train_ratio)))
    cutoff_period = unique_periods[cutoff_idx - 1]

    train_mask = period_key <= cutoff_period
    return (
        ordered.loc[train_mask].reset_index(drop=True),
        ordered.loc[~train_mask].reset_index(drop=True),
    )


class DataPreprocessor:
    def __init__(self, file_path: str | None = None, df: pd.DataFrame | None = None):
        self.file_path = file_path
        if df is not None:
            self.df = df.copy()
        elif file_path is not None:
            self.df = load_data(file_path)
        else:
            self.df = None

    def preprocess(self) -> pd.DataFrame:
        df = self.df.copy()

        # Recover year/month from month_year if needed (compatibility with raw CSVs)
        if ("year" not in df.columns or "month" not in df.columns) and "month_year" in df.columns:
            parsed = pd.to_datetime(df["month_year"], errors="coerce")
            df["year"] = parsed.dt.year
            df["month"] = parsed.dt.month

        # Drop raw string timestamp identifier after extracting year & month
        df.drop(columns=["month_year"], inplace=True, errors="ignore")

        # Handle missing competitor prices conservatively with own unit_price
        comp_cols = ["comp_1", "comp_2", "comp_3"]
        for col in comp_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df["unit_price"])

        # One-hot encode product_category_name instead of dropping it
        if "product_category_name" in df.columns:
            df = pd.get_dummies(
                df, columns=["product_category_name"], prefix="cat", dtype=int
            )

        df.dropna(inplace=True)
        self.df = df
        return df

    def feature_engineering(self) -> pd.DataFrame:
        df = self.df.copy()

        # Competitor price differentials & relative price ratio
        comp_avg = (df["comp_1"] + df["comp_2"] + df["comp_3"]) / 3.0
        df["price_diff_1"] = df["unit_price"] - df["comp_1"]
        df["price_diff_2"] = df["unit_price"] - df["comp_2"]
        df["price_diff_3"] = df["unit_price"] - df["comp_3"]
        df["price_ratio"] = df["unit_price"] / comp_avg.clip(lower=1e-3)

        if "qty" in df.columns:
            df["revenue"] = df["unit_price"] * df["qty"]

        self.df = df
        return df

    def run(self) -> pd.DataFrame:
        self.preprocess()
        return self.feature_engineering()

    def save_processed(self, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.df.to_csv(output_path, index=False)
        print(f"✅ Processed data saved to {output_path}")


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(root, "data", "raw", "retail_price.csv")
    output_path = os.path.join(root, "data", "processed", "processed_retail_price.csv")

    preprocessor = DataPreprocessor(input_path)
    if preprocessor.df is not None:
        preprocessor.run()
        preprocessor.save_processed(output_path)
        print(f"✅ Final Processed Data Shape: {preprocessor.df.shape} (Rows, Columns)")
