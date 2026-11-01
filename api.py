"""
FastAPI service for the Dynamic Pricing Optimizer.

Endpoints:
- GET  /health           -> service & model readiness check
- GET  /products         -> list available SKU IDs
- GET  /model/info       -> model metadata, training metrics, and monotonic constraints
- GET  /backtest/summary -> out-of-time policy backtest KPIs
- POST /optimize         -> compute optimal price, expected demand, profit, and lift
"""
from __future__ import annotations

from contextlib import asynccontextmanager
import json
import os
import sys
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from optimization import PriceOptimizer

MODEL_PATH = os.path.join(ROOT, "models", "best_model.pkl")
DATA_PATH = os.path.join(ROOT, "data", "processed", "processed_retail_price.csv")
TRAIN_REPORT_PATH = os.path.join(ROOT, "models", "training_report.json")
BACKTEST_REPORT_PATH = os.path.join(ROOT, "models", "backtest_report.json")

optimizer: Optional[PriceOptimizer] = None


def _ensure_loaded() -> PriceOptimizer:
    global optimizer
    if optimizer is None and os.path.exists(MODEL_PATH) and os.path.exists(DATA_PATH):
        optimizer = PriceOptimizer(MODEL_PATH, DATA_PATH)
    if optimizer is None:
        raise HTTPException(status_code=503, detail="Optimizer not loaded")
    return optimizer


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global optimizer
    if os.path.exists(MODEL_PATH) and os.path.exists(DATA_PATH):
        optimizer = PriceOptimizer(MODEL_PATH, DATA_PATH)
    yield


app = FastAPI(
    title="Dynamic Pricing API",
    version="1.1.0",
    description="REST API for demand forecasting & profit-maximizing price optimization",
    lifespan=lifespan,
)


class OptimizeRequest(BaseModel):
    product_id: str = Field(..., description="Product ID to optimize for")
    competitor_price: float = Field(..., gt=0, description="Competitor price")
    product_cost: float = Field(0.0, ge=0, description="Unit cost (COGS)")
    month: int = Field(6, ge=1, le=12, description="Target month (1-12) for seasonality")


class OptimizeResponse(BaseModel):
    product_id: str
    competitor_price: float
    product_cost: float
    current_price: float
    best_price: float
    best_qty: float
    best_revenue: float
    best_profit: Optional[float]
    baseline_qty: float
    baseline_revenue: float
    baseline_profit: Optional[float]
    lift_pct: float
    lift_qty: float
    optimal_by: str


class ProductListResponse(BaseModel):
    products: list[str]


class ModelInfoResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_type: str
    n_features: int
    features: list[str]
    metrics: dict


@app.get("/health")
def health():
    try:
        _ensure_loaded()
        loaded = True
    except Exception:
        loaded = False
    return {"status": "ok" if loaded else "degraded", "optimizer_loaded": loaded}


@app.get("/products", response_model=ProductListResponse)
def list_products():
    opt = _ensure_loaded()
    return {"products": opt.products}


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    opt = _ensure_loaded()
    metrics_payload: dict = {}
    if os.path.exists(TRAIN_REPORT_PATH):
        with open(TRAIN_REPORT_PATH, "r", encoding="utf-8") as f:
            report = json.load(f)
            metrics_payload = {
                "best_model": report.get("best_model"),
                "train_period": report.get("train_period"),
                "test_period": report.get("test_period"),
                "metrics": report.get("metrics", {}),
                "monotone_constraints": report.get("monotone_constraints", {}),
            }
    return {
        "model_type": type(opt.model).__name__,
        "n_features": len(opt.model_features),
        "features": opt.model_features,
        "metrics": metrics_payload,
    }


@app.get("/backtest/summary")
def backtest_summary():
    if not os.path.exists(BACKTEST_REPORT_PATH):
        raise HTTPException(status_code=404, detail="Backtest report not found")
    with open(BACKTEST_REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)
    return {
        "method": report.get("method"),
        "n_products_evaluated": report.get("n_products_evaluated"),
        "aggregate": report.get("aggregate"),
    }


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest):
    opt = _ensure_loaded()

    if req.product_id not in opt.products:
        raise HTTPException(status_code=404, detail=f"Product {req.product_id} not found")

    summary, _ = opt.get_optimal_price(
        competitor_price=req.competitor_price,
        product_cost=req.product_cost,
        product_id=req.product_id,
        month=req.month,
    )

    lift_key = f"{summary['optimal_by']}_pct"
    return {
        "product_id": summary["product_id"],
        "competitor_price": summary["competitor_price"],
        "product_cost": summary["product_cost"],
        "current_price": summary["current_price"],
        "best_price": summary["best_price"],
        "best_qty": summary["best_qty"],
        "best_revenue": summary["best_revenue"],
        "best_profit": summary["best_profit"],
        "baseline_qty": summary["baseline_qty"],
        "baseline_revenue": summary["baseline_revenue"],
        "baseline_profit": summary["baseline_profit"],
        "lift_pct": summary["lift"][lift_key],
        "lift_qty": summary["lift"]["qty_units"],
        "optimal_by": summary["optimal_by"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)