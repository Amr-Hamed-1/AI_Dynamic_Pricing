"""
Synthetic data generator for the Dynamic Pricing project.

Generates a realistic e-commerce panel (product x month) across two full annual
cycles (2017-01 .. 2018-12) with an explicit, interpretable price-elasticity
relationship: units sold respond to how the product price compares to its
competitors' average, category demand, and Q4 seasonality.

The raw schema matches the Olist `retail_price.csv` benchmark dataset so the
pipeline works seamlessly on both synthetic and real data.

Run:
    python src/generate_data.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import DataPreprocessor

PRODUCT_CATEGORIES = [
    "bed_bath_table",
    "garden_tools",
    "consoles_games",
    "health_beauty",
    "cool_stuff",
    "perfumery",
    "computers_accessories",
    "watches_gifts",
    "furniture_decor",
]

CATEGORY_PRICE_RANGE = {
    "bed_bath_table": (50, 250),
    "garden_tools": (30, 200),
    "consoles_games": (150, 360),
    "health_beauty": (20, 120),
    "cool_stuff": (25, 160),
    "perfumery": (40, 180),
    "computers_accessories": (60, 350),
    "watches_gifts": (45, 300),
    "furniture_decor": (80, 400),
}

CATEGORY_BASE_DEMAND = {
    "bed_bath_table": 1.0,
    "garden_tools": 0.9,
    "consoles_games": 1.4,
    "health_beauty": 1.1,
    "cool_stuff": 1.2,
    "perfumery": 1.0,
    "computers_accessories": 1.3,
    "watches_gifts": 0.8,
    "furniture_decor": 0.7,
}


PRICE_ELASTICITY = 1.8  # true constant price elasticity of demand (|dlnQ/dlnP|)


def _clip(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def true_expected_demand(category: str, volume, score, month, price, comp_avg):
    """Ground-truth expected demand (noise-free) of the synthetic market.

    Used by the generator to create `qty`, and by `backtest.py` to score the
    optimizer's recommendations against reality instead of against the model
    itself (which would be circular). Works on scalars or numpy arrays.
    """
    price = np.asarray(price, dtype=float)
    base_demand = (
        CATEGORY_BASE_DEMAND[category]
        * 14.0
        * (np.asarray(volume, dtype=float) / 10000.0) ** 0.35
        * (np.asarray(score, dtype=float) / 4.0) ** 0.5
    )
    season = np.where(np.isin(month, (11, 12)), 1.35, 1.0)
    log_ratio = np.clip(np.log(price / np.asarray(comp_avg, dtype=float)), -0.7, 0.7)
    return base_demand * season * np.exp(-PRICE_ELASTICITY * log_ratio)


def generate_frame(
    n_products: int = 200, months: int = 24, seed: int = 42
) -> pd.DataFrame:
    """Create a chronological panel of `n_products` across `months` starting 2017-01."""
    rng = np.random.default_rng(seed)
    n_products = max(n_products, 10)

    # Strictly sequential calendar months (e.g. 2017-01 through 2018-12)
    calendar = pd.date_range("2017-01-01", periods=months, freq="MS")

    rows: list[dict] = []

    for p in range(n_products):
        category = str(rng.choice(PRODUCT_CATEGORIES))
        price_lo, price_hi = CATEGORY_PRICE_RANGE[category]

        base_price = round(float(rng.uniform(price_lo, price_hi)), 2)
        weight = int(_clip(rng.integers(100, 9800), 100, 9800))
        volume = int(_clip(rng.integers(600, 33000), 600, 33000))
        name_len = int(_clip(rng.integers(29, 61), 29, 60))
        desc_len = int(_clip(rng.integers(100, 3100), 100, 3006))
        photos = int(_clip(rng.integers(1, 9), 1, 8))
        score = round(_clip(rng.normal(4.1, 0.15), 3.3, 4.5), 1)

        # Track previous month's price strictly per-product (no cross-product leakage)
        prev_price: float | None = None

        for dt in calendar:
            year = int(dt.year)
            month = int(dt.month)

            comp_1 = round(_clip(base_price * rng.uniform(0.85, 1.25), 19.9, 360), 2)
            comp_2 = round(_clip(base_price * rng.uniform(0.85, 1.25), 19.9, 360), 2)
            comp_3 = round(_clip(base_price * rng.uniform(0.85, 1.25), 19.9, 360), 2)

            unit_price = round(_clip(base_price * rng.uniform(0.8, 1.2), 19.9, 360), 2)
            lag_price = (
                prev_price
                if prev_price is not None
                else round(_clip(base_price * rng.uniform(0.98, 1.02), 19.9, 360), 2)
            )
            prev_price = unit_price

            ps1 = round(_clip(rng.normal(4.1, 0.1), 3.7, 4.5), 1)
            ps2 = round(_clip(rng.normal(4.1, 0.15), 3.3, 4.5), 1)
            ps3 = round(_clip(rng.normal(4.0, 0.15), 3.3, 4.4), 1)
            fp1 = round(float(rng.uniform(0.1, 57.0)), 2)
            fp2 = round(float(rng.uniform(4.4, 57.0)), 2)
            fp3 = round(float(rng.uniform(7.6, 57.0)), 2)

            holiday = int(rng.choice([0, 1, 2, 3, 4], p=[0.5, 0.2, 0.15, 0.1, 0.05]))
            weekend = int(_clip(int(rng.choice([8, 9, 10])), 8, 10))
            weekday = int(_clip(21 + int(rng.normal(0, 1.0)), 19, 23))
            customers_base = _clip(rng.integers(30, 120), 1, 340)

            # --- Interpretable demand model (single source of truth) ---------
            comp_avg = (comp_1 + comp_2 + comp_3) / 3.0
            mean_qty = float(
                true_expected_demand(category, volume, score, month, unit_price, comp_avg)
            )

            # Realistic demand noise (8% log-normal standard deviation)
            noise = rng.lognormal(0, 0.08)

            qty = int(max(1, round(mean_qty * noise)))

            # --- Realized post-decision market outcomes ---------------------
            freight_price = round(float(rng.uniform(5, 30)), 2)
            total_price = round(unit_price * qty + freight_price, 2)
            # Note: `customers` is a contemporaneous outcome (dropped during training)
            customers = int(_clip(customers_base + qty * rng.uniform(0.8, 1.4), 1, 400))
            s = round(float(rng.uniform(0.5, 100.0)), 2)
            month_year = f"{year}-{month:02d}"

            rows.append({
                "product_id": f"prod_{p:04d}",
                "product_category_name": category,
                "month_year": month_year,
                "qty": qty,
                "total_price": total_price,
                "freight_price": freight_price,
                "unit_price": unit_price,
                "product_name_lenght": name_len,
                "product_description_lenght": desc_len,
                "product_photos_qty": photos,
                "product_weight_g": weight,
                "product_score": score,
                "customers": customers,
                "weekday": weekday,
                "weekend": weekend,
                "holiday": holiday,
                "month": month,
                "year": year,
                "s": s,
                "volume": volume,
                "comp_1": comp_1,
                "ps1": ps1,
                "fp1": fp1,
                "comp_2": comp_2,
                "ps2": ps2,
                "fp2": fp2,
                "comp_3": comp_3,
                "ps3": ps3,
                "fp3": fp3,
                "lag_price": lag_price,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(root, "data", "raw", "retail_price.csv")
    proc_path = os.path.join(root, "data", "processed", "processed_retail_price.csv")

    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    os.makedirs(os.path.dirname(proc_path), exist_ok=True)

    print("Generating synthetic data...")
    raw = generate_frame(n_products=200, months=24)
    raw.to_csv(raw_path, index=False)
    print(f"Raw       -> {raw.shape[0]} rows x {raw.shape[1]} cols -> {raw_path}")

    # Single source of truth: run DataPreprocessor from src/preprocessing.py
    preprocessor = DataPreprocessor(df=raw)
    proc = preprocessor.run()
    preprocessor.save_processed(proc_path)
    print(f"Processed -> {proc.shape[0]} rows x {proc.shape[1]} cols -> {proc_path}")

    comp_avg = proc[["comp_1", "comp_2", "comp_3"]].mean(axis=1)
    cheaper = proc[proc["unit_price"] < comp_avg]["qty"].mean()
    dearer = proc[proc["unit_price"] >= comp_avg]["qty"].mean()
    print("\nPrice-elasticity sanity check:")
    print(f"  avg qty when cheaper than competitors : {cheaper:7.1f}")
    print(f"  avg qty when dearer  than competitors : {dearer:7.1f}")
