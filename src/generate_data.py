"""
Synthetic data generator for the Dynamic Pricing project.

Generates a realistic e-commerce panel (product x month) with an explicit,
interpretable *price-elasticity* relationship: the number of units sold
responds to how the product price compares to its competitors' average.

The schema matches the original Olist-style `retail_price.csv` so the rest of
the pipeline (preprocessing -> training -> optimization) works unchanged.

Run:
    python src/generate_data.py
Output:
    data/raw/retail_price.csv          (same schema as before)
    data/processed/processed_retail_price.csv  (fully featured, ready for training)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

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

# (min_price, max_price) per category so products live in realistic bands
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

# Baseline popularity multiplier per category (higher -> more inherent demand)
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


def _clip(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def generate_frame(n_products: int = 200, months: int = 20,
                   seed: int = 42) -> pd.DataFrame:
    """Create a synthetic panel of `n_products` across `months` (2017-2018)."""
    rng = np.random.default_rng(seed)
    n_products = max(n_products, 10)

    years = np.full(months, 2017)
    years[months // 2:] = 2018
    months_idx = np.arange(1, months + 1) % 12
    months_idx[months_idx == 0] = 12

    rows: list[dict] = []

    for p in range(n_products):
        category = rng.choice(PRODUCT_CATEGORIES)
        price_lo, price_hi = CATEGORY_PRICE_RANGE[category]

        # A product's own "locked-in" price varies a little over time (lag) but
        # stays close to its base level -> mimics a stable catalogue price.
        base_price = round(rng.uniform(price_lo, price_hi), 2)
        weight = _clip(rng.integers(100, 9800), 100, 9800)
        volume = _clip(rng.integers(600, 33000), 600, 33000)
        name_len = _clip(rng.integers(29, 61), 29, 60)
        desc_len = _clip(rng.integers(100, 3100), 100, 3006)
        photos = _clip(rng.integers(1, 9), 1, 8)
        score = round(_clip(rng.normal(4.1, 0.15), 3.3, 4.5), 1)
        seasonality = category + "_" if False else None  # keep code simple
        del seasonality

        for m_idx, month in enumerate(months_idx):
            # Competitors price a bit around our base price
            comp_1 = round(_clip(base_price * rng.uniform(0.85, 1.25), 19.9, 360), 2)
            comp_2 = round(_clip(base_price * rng.uniform(0.85, 1.25), 19.9, 360), 2)
            comp_3 = round(_clip(base_price * rng.uniform(0.85, 1.25), 19.9, 360), 2)

            # Our chosen price for this month (the decision variable)
            unit_price = round(_clip(base_price * rng.uniform(0.8, 1.2), 19.9, 360), 2)

            lag_price = round(
                _clip(rows[-1]["unit_price"] if rows else base_price
                      * rng.uniform(0.98, 1.02), 19.9, 360), 2
            ) if rows else unit_price

            # Promo flags & freight (roughly like the original features)
            ps1 = round(_clip(rng.normal(4.1, 0.1), 3.7, 4.5), 1)
            ps2 = round(_clip(rng.normal(4.1, 0.15), 3.3, 4.5), 1)
            ps3 = round(_clip(rng.normal(4.0, 0.15), 3.3, 4.4), 1)
            fp1 = round(rng.uniform(0.1, 57.0), 2)
            fp2 = round(rng.uniform(4.4, 57.0), 2)
            fp3 = round(rng.uniform(7.6, 57.0), 2)

            # Month / holiday / weekend business-day context
            holiday = int(rng.choice([0, 1, 2, 3, 4], p=[0.5, 0.2, 0.15, 0.1, 0.05]))
            weekend = _clip(int(rng.choice([8, 9, 10])), 8, 10)
            weekday = _clip(month * 2 - 1 + int(rng.normal(0, 1.5)), 20, 23)
            customers_base = _clip(rng.integers(30, 120), 1, 340)

            # --- Interpretable demand model ---------------------------------
            comp_avg = (comp_1 + comp_2 + comp_3) / 3.0
            price_ratio = unit_price / comp_avg  # >1 -> dearer, <1 -> cheaper

            # demand baseline driven by category + product size + season
            base_demand = (CATEGORY_BASE_DEMAND[category]
                           * 14.0
                           * (volume / 10000.0) ** 0.35)

            # seasonal bump in Nov/Dec
            season = 1.0 + (0.35 if month in (11, 12) else 0.0)

            # price-elasticity term (logarithmic around the competitor mean)
            elasticity = np.clip(np.log(price_ratio), -0.7, 0.7)
            price_term = np.exp(-1.8 * elasticity)

            noise = rng.lognormal(0, 0.20)

            expected_qty = base_demand * season * price_term * noise
            qty = int(max(1, round(expected_qty)))

            # --- Derived / market columns -----------------------------------
            freight_price = round(rng.uniform(5, 30), 2)
            total_price = round(unit_price * qty + freight_price, 2)
            customers = int(_clip(customers_base
                                  + qty * rng.uniform(0.8, 1.4), 1, 400))
            s = round(rng.uniform(0.5, 100.0), 2)  # market share proxy
            year = int(years[m_idx])
            month_year = f"{year}-{month:02d}" if False else f"{year}-{month:02d}"

            rows.append({
                "product_id": f"prod_{p:04d}",
                "product_category_name": category,
                "month_year": month_year,
                "qty": qty,
                "total_price": total_price,
                "freight_price": freight_price,
                "unit_price": unit_price,
                "product_name_lenght": int(name_len),
                "product_description_lenght": int(desc_len),
                "product_photos_qty": int(photos),
                "product_weight_g": int(weight),
                "product_score": score,
                "customers": customers,
                "weekday": int(weekday),
                "weekend": int(weekend),
                "holiday": holiday,
                "month": int(month),
                "year": year,
                "s": s,
                "volume": int(volume),
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

    cols = ["product_id", "product_category_name", "month_year", "qty",
            "total_price", "freight_price", "unit_price", "product_name_lenght",
            "product_description_lenght", "product_photos_qty",
            "product_weight_g", "product_score", "customers", "weekday",
            "weekend", "holiday", "month", "year", "s", "volume",
            "comp_1", "ps1", "fp1", "comp_2", "ps2", "fp2",
            "comp_3", "ps3", "fp3", "lag_price"]
    return pd.DataFrame(rows, columns=cols)


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror src/preprocessing.py so data/processed is created here too."""
    comp_cols = ["comp_1", "comp_2", "comp_3"]
    for col in comp_cols:
        df[col] = df[col].fillna(df["unit_price"])

    # Keep product_id so downstream optimisation can operate per-product.
    work = df.drop(columns=["product_category_name", "month_year"],
                   errors="ignore").dropna()

    work["price_diff_1"] = work["unit_price"] - work["comp_1"]
    work["price_diff_2"] = work["unit_price"] - work["comp_2"]
    work["price_diff_3"] = work["unit_price"] - work["comp_3"]
    work["revenue"] = work["unit_price"] * work["qty"]
    return work


if __name__ == "__main__":
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(root, "data", "raw", "retail_price.csv")
    proc_path = os.path.join(root, "data", "processed",
                             "processed_retail_price.csv")

    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    os.makedirs(os.path.dirname(proc_path), exist_ok=True)

    print("Generating synthetic data...")
    raw = generate_frame(n_products=200, months=20)
    raw.to_csv(raw_path, index=False)
    print(f"Raw     -> {raw.shape[0]} rows x {raw.shape[1]} cols -> {raw_path}")

    proc = _preprocess(raw)
    proc.to_csv(proc_path, index=False)
    print(f"Processed -> {proc.shape[0]} rows x {proc.shape[1]} cols -> {proc_path}")

    # quick sanity check on the price-elasticity signal
    comp_avg = proc[["comp_1", "comp_2", "comp_3"]].mean(axis=1)
    cheaper = proc[proc["unit_price"] < comp_avg]["qty"].mean()
    dearer = proc[proc["unit_price"] >= comp_avg]["qty"].mean()
    print(f"\nPrice-elasticity sanity check:")
    print(f"  avg qty when cheaper than competitors : {cheaper:7.1f}")
    print(f"  avg qty when dearer  than competitors : {dearer:7.1f}")
