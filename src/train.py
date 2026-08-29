"""
Training pipeline for the Dynamic Pricing project.

- Loads processed data
- Splits on time (temporal split): first 80% months = train, last 20% = test.
  This mimics how the model would be used in production (predict future demand).
- Trains RandomForest & XGBoost, tunes XGBoost with cross-validation
- Selects the best model by RMSE, saves it and writes a small report.

Run:
    python src/train.py
"""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

TARGET = "qty"
# Columns that are either the target, leak the target, or shouldn't be features
COLUMNS_TO_DROP = [
    "qty", "revenue", "total_price", "freight_price",
    # 'month_year'/'product_id' are identifiers, not predictors
]
IDENTITY_COLS = ["product_id", "month_year"]

# Time split ratio (train / whole)
TRAIN_RATIO = 0.8


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def train_model() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root, "data", "processed",
                             "processed_retail_price.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data not found at {data_path}")

    data = pd.read_csv(data_path)
    data = data.sort_values(by=["month", "year"]).reset_index(drop=True)

    X = data.drop(columns=COLUMNS_TO_DROP, errors="ignore")
    y = data[TARGET]

    # Temporal split: train on earlier rows, test on later rows
    split_idx = int(len(X) * TRAIN_RATIO)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Avoid leaking the target through identity columns; the trainer should not
    # see product_id as a numeric feature (it has no predictive ordering).
    present_identity = [c for c in IDENTITY_COLS if c in X_train.columns]
    identity_train = X_train[present_identity].copy()
    identity_test = X_test[present_identity].copy()
    X_train = X_train.drop(columns=present_identity, errors="ignore")
    X_test = X_test.drop(columns=present_identity, errors="ignore")

    print(f"🚀 Temporal split -> train: {len(X_train)}, test: {len(X_test)}")

    # ---- Baseline: Random Forest -----------------------------------------
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42,
                                     n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    # ---- Main model: XGBoost with light hyperparameter tuning ------------
    xgb = XGBRegressor(random_state=42, n_jobs=-1, objective="reg:squarederror")
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
    }
    grid = GridSearchCV(
        xgb, param_grid, cv=5, scoring="neg_root_mean_squared_error",
        n_jobs=-1, verbose=0,
    )
    grid.fit(X_train, y_train)
    xgb_model = grid.best_estimator_
    xgb_preds = xgb_model.predict(X_test)

    # ---- Compare & save best ---------------------------------------------
    models = {
        "RandomForest": (rf_model, rf_preds, rf_model.feature_names_in_),
        "XGBoost": (xgb_model, xgb_preds, xgb_model.feature_names_in_),
    }
    metrics = {}
    for name, (model, preds, features) in models.items():
        metrics[name] = {
            "MAE": float(mean_absolute_error(y_test, preds)),
            "RMSE": _rmse(y_test, preds),
            "R2": float(r2_score(y_test, preds)),
        }
        print(f"{name:<14} MAE={metrics[name]['MAE']:.3f} "
              f"RMSE={metrics[name]['RMSE']:.3f} R²={metrics[name]['R2']:.3f}")

    best_name = min(metrics, key=lambda k: metrics[k]["RMSE"])
    best_model, best_preds, best_features = models[best_name]
    print(f"\n🏆 Best model: {best_name}")

    os.makedirs(os.path.join(root, "models"), exist_ok=True)
    model_path = os.path.join(root, "models", "best_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"✅ Model saved to {model_path}")

    # ---- Write a report (metrics + feature importances) ------------------
    report = {
        "best_model": best_name,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "metrics": metrics,
        "feature_importance": _feature_importance(best_model, best_features),
    }
    report_path = os.path.join(root, "models", "training_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"✅ Report written to {report_path}")


def _feature_importance(model, feature_names):
    try:
        values = model.feature_importances_
        names = list(feature_names)
    except (AttributeError, TypeError):
        # XGBoost booster fallback
        imp = model.get_booster().get_score(importance_type="gain")
        names = list(imp.keys())
        values = list(imp.values())
    order = np.argsort(values)[::-1]
    return [
        {"feature": names[int(i)], "importance": float(values[int(i)])}
        for i in order[:20]
    ]


if __name__ == "__main__":
    train_model()
