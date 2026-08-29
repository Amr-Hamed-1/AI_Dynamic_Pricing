"""
FastAPI wrapper for the Dynamic Pricing Optimizer.

Endpoints:
- GET  /health                 -> health check
- POST /optimize               -> run price optimization for a product
- GET  /products               -> list available product IDs
- GET  /model/info             -> model metadata (metrics, features)
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure src is on path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from optimization import PriceOptimizer

app = FastAPI(
    title="Dynamic Pricing API",
    version="1.0.0",
    description="REST API for price optimization & demand forecasting",
)

# Load optimizer once at startup
ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "models", "best_model.pkl")
DATA_PATH = os.path.join(ROOT, "data", "processed", "processed_retail_price.csv")

optimizer: Optional[PriceOptimizer] = None


@app.on_event("startup")
def load_optimizer():
    global optimizer
    if os.path.exists(MODEL_PATH) and os.path.exists(DATA_PATH):
        optimizer = PriceOptimizer(MODEL_PATH, DATA_PATH)
    else:
        # In Docker, data/model are generated at build time, so they should exist.
        # If not, optimizer stays None and endpoints will 503.
        pass


# ---------------------------
# Schemas
# ---------------------------
class OptimizeRequest(BaseModel):
    product_id: str = Field(..., description="Product ID to optimize for")
    competitor_price: float = Field(..., gt=0, description="Competitor price")
    product_cost: float = Field(0.0, ge=0, description="Product cost (COGS)")
    month: int = Field(6, ge=1, le=12, description="Month for seasonality context")


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
    model_type: str
    n_features: int
    features: list[str]
    metrics: dict


# ---------------------------
# Endpoints
# ---------------------------
@app.get("/health")
def health():
    return {"status": "ok", "optimizer_loaded": optimizer is not None}


@app.get("/products", response_model=ProductListResponse)
def list_products():
    if optimizer is None:
        raise HTTPException(503, "Optimizer not loaded")
    return {"products": optimizer.products}


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    if optimizer is None:
        raise HTTPException(503, "Optimizer not loaded")
    try:
        features = list(optimizer.model.feature_names_in_)
    except AttributeError:
        features = list(optimizer.model.get_booster().feature_names)
    return {
        "model_type": type(optimizer.model).__name__,
        "n_features": len(features),
        "features": features,
        "metrics": {"note": "See models/training_report.json for full metrics"},
    }


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest):
    if optimizer is None:
        raise HTTPException(503, "Optimizer not loaded")

    if req.product_id not in optimizer.products:
        raise HTTPException(404, f"Product {req.product_id} not found")

    summary, _ = optimizer.get_optimal_price(
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


# ---------------------------
# Entrypoint for `uvicorn api:app --host 0.0.0.0 --port 8000`
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)