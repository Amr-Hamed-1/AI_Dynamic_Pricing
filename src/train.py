"""
Training pipeline for the Dynamic Pricing project.

- Loads processed panel data
- Splits strictly chronologically by (year, month) via `temporal_train_test_split`
- Drops contemporaneous post-decision columns (`customers`, `revenue`, `total_price`, `freight_price`)
  to prevent target leakage
- Tunes XGBoost with `TimeSeriesSplit` cross-validation and economic **monotonic constraints**
  (higher own price / price gap vs. competitors cannot increase predicted demand)
- Compares against RandomForest baseline, saves best model + JSON report + MLflow run.

Run:
    python src/train.py
"""
from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from src.preprocessing import temporal_train_test_split

try:  # pragma: no cover - environment dependent
    import mlflow
    _HAS_MLFLOW = True
except ImportError:  # pragma: no cover
    mlflow = None
    _HAS_MLFLOW = False

TARGET = "qty"

# Columns that either are the target or leak post-decision outcomes (`customers`, `revenue`, `total_price`)
COLUMNS_TO_DROP = [
    "qty",
    "revenue",
    "total_price",
    "freight_price",
    "customers",
]
IDENTITY_COLS = ["product_id", "month_year"]

# Features enforced to have non-positive (-1) monotonic effect on demand (`qty`)
NEGATIVE_PRICE_FEATURES = {
    "unit_price": -1,
    "price_diff_1": -1,
    "price_diff_2": -1,
    "price_diff_3": -1,
    "price_ratio": -1,
}

TRAIN_RATIO = 0.8


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def train_model() -> dict:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root, "data", "processed", "processed_retail_price.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data not found at {data_path}")

    data = pd.read_csv(data_path)

    # Strictly chronological split by (year, month)
    train_df, test_df = temporal_train_test_split(data, train_ratio=TRAIN_RATIO)

    train_span = (
        f"{int(train_df['year'].iloc[0])}-{int(train_df['month'].iloc[0]):02d} .. "
        f"{int(train_df['year'].iloc[-1])}-{int(train_df['month'].iloc[-1]):02d}"
    )
    test_span = (
        f"{int(test_df['year'].iloc[0])}-{int(test_df['month'].iloc[0]):02d} .. "
        f"{int(test_df['year'].iloc[-1])}-{int(test_df['month'].iloc[-1]):02d}"
    )

    drop_all = COLUMNS_TO_DROP + IDENTITY_COLS
    X_train = train_df.drop(columns=drop_all, errors="ignore")
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=drop_all, errors="ignore")
    y_test = test_df[TARGET]

    print(
        f"🚀 Chronological split -> train: {len(X_train)} rows ({train_span}), "
        f"test: {len(X_test)} rows ({test_span})"
    )

    # ---- Baseline: Random Forest -----------------------------------------
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    # ---- Main model: XGBoost + TimeSeriesSplit + Monotonic Constraints ---
    # Demand is multiplicative in price (constant-elasticity), so the model is
    # trained on log1p(qty). This yields a smoother price-response curve that the
    # optimizer's argmax can't exploit (raw-target trees have step artifacts that
    # produced *negative* ground-truth lift in the backtest). Predictions are
    # returned in units via expm1, so callers are unaffected.
    monotone_map = {
        col: constraint
        for col, constraint in NEGATIVE_PRICE_FEATURES.items()
        if col in X_train.columns
    }
    xgb = TransformedTargetRegressor(
        regressor=XGBRegressor(
            random_state=42,
            n_jobs=-1,
            objective="reg:squarederror",
            monotone_constraints=monotone_map,
        ),
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )
    param_grid = {
        "regressor__n_estimators": [200, 400],
        "regressor__max_depth": [4, 6],
        "regressor__learning_rate": [0.05, 0.1],
        "regressor__subsample": [0.8],
        "regressor__colsample_bytree": [0.8],
    }
    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(
        xgb,
        param_grid,
        cv=tscv,
        scoring="neg_root_mean_squared_error",  # scored in units (after expm1)
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    xgb_model = grid.best_estimator_
    xgb_preds = xgb_model.predict(X_test)

    # ---- Compare & save best ---------------------------------------------
    models = {
        "RandomForest": (rf_model, rf_preds, rf_model.feature_names_in_),
        "XGBoost": (xgb_model, xgb_preds, list(X_train.columns)),
    }
    metrics = {}
    for name, (model, preds, features) in models.items():
        metrics[name] = {
            "MAE": float(mean_absolute_error(y_test, preds)),
            "RMSE": _rmse(y_test, preds),
            "R2": float(r2_score(y_test, preds)),
        }
        print(
            f"{name:<14} MAE={metrics[name]['MAE']:.3f} "
            f"RMSE={metrics[name]['RMSE']:.3f} R²={metrics[name]['R2']:.3f}"
        )

    best_name = min(metrics, key=lambda k: metrics[k]["RMSE"])
    best_model, _, best_features = models[best_name]
    print(f"\n🏆 Best model: {best_name} (best params: {grid.best_params_})")

    os.makedirs(os.path.join(root, "models"), exist_ok=True)
    model_path = os.path.join(root, "models", "best_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"✅ Model saved to {model_path}")

    report = {
        "best_model": best_name,
        "best_params": grid.best_params_ if best_name == "XGBoost" else {},
        "monotone_constraints": monotone_map if best_name == "XGBoost" else {},
        "target_transform": "log1p(qty), predictions returned via expm1" if best_name == "XGBoost" else "none",
        "train_period": train_span,
        "test_period": test_span,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "metrics": metrics,
        "feature_importance": _feature_importance(best_model, best_features),
    }
    report_path = os.path.join(root, "models", "training_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"✅ Report written to {report_path}")

    if _HAS_MLFLOW:
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{os.path.join(root, 'mlflow.db')}")
        )
        mlflow.set_experiment("dynamic-pricing")
        with mlflow.start_run(run_name=f"{best_name}-chronological-split"):
            for name, m in metrics.items():
                mlflow.log_metric(f"{name}_MAE", m["MAE"])
                mlflow.log_metric(f"{name}_RMSE", m["RMSE"])
                mlflow.log_metric(f"{name}_R2", m["R2"])
            mlflow.log_params({
                "split": "chronological_year_month",
                "cv": "TimeSeriesSplit(5)",
                "monotone_constraints": bool(monotone_map),
                "train_ratio": TRAIN_RATIO,
                "train_period": train_span,
                "test_period": test_span,
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
                "best_model": best_name,
            })
            mlflow.log_artifact(model_path, artifact_path="model")
            mlflow.log_artifact(report_path, artifact_path="report")
            print(
                "✅ Experiment logged to MLflow -> run "
                f"'{mlflow.active_run().info.run_id}'"
            )

    return report


def _feature_importance(model, feature_names):
    model = getattr(model, "regressor_", model)  # unwrap TransformedTargetRegressor
    try:
        values = model.feature_importances_
        names = list(feature_names)
    except (AttributeError, TypeError):
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
