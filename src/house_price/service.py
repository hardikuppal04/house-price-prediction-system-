"""FastAPI inference service with safe loading and operational telemetry."""

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from house_price.config import load_config
from house_price.inflation import CPI_U_REFERENCE, adjust_2010_dollars
from house_price.production import load_production_model, predict_one
from house_price.schema import PredictionRequest

logger = logging.getLogger("house_price.api")
REQUESTS = Counter("house_price_requests_total", "API requests", ["path", "status"])
LATENCY = Histogram("house_price_request_seconds", "Request latency", ["path"])
PREDICTIONS = Histogram(
    "house_price_prediction_dollars",
    "Historical predicted price",
    buckets=(50_000, 100_000, 150_000, 200_000, 300_000, 500_000, 1_000_000),
)


class PredictionResponse(BaseModel):
    predicted_price_2010_usd: float
    interval_90_low: float
    interval_90_high: float
    inflation_adjusted_reference_usd: float
    inflation_adjusted_interval_90_low: float
    inflation_adjusted_interval_90_high: float
    inflation_reference_year: int
    inflation_factor: float
    inflation_note: str
    model_version: str
    warnings: list[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    try:
        app.state.model, app.state.metadata = load_production_model(cfg.paths.models)
        app.state.ready_error = None
    except Exception as exc:  # service stays live but correctly reports not-ready
        app.state.model, app.state.metadata = None, None
        app.state.ready_error = type(exc).__name__
        logger.error(json.dumps({"event": "model_load_failed", "error_type": type(exc).__name__}))
    yield


app = FastAPI(
    title="Historical Ames House Price API",
    version="1.0.0",
    description="Historical deployment simulation. Predictions are approximately 2010 dollars.",
    lifespan=lifespan,
)
app.state.model = None
app.state.metadata = None
app.state.ready_error = "startup_not_completed"


@app.middleware("http")
async def operations_middleware(request: Request, call_next):
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        logger.exception(
            json.dumps(
                {
                    "event": "request_failed",
                    "correlation_id": correlation_id,
                    "path": request.url.path,
                }
            )
        )
        response = Response("Internal server error", status_code=500)
    elapsed = time.perf_counter() - started
    response.headers["X-Correlation-ID"] = correlation_id
    REQUESTS.labels(request.url.path, str(status)).inc()
    LATENCY.labels(request.url.path).observe(elapsed)
    logger.info(
        json.dumps(
            {
                "event": "request",
                "correlation_id": correlation_id,
                "path": request.url.path,
                "status": status,
                "duration_ms": round(elapsed * 1000, 2),
            }
        )
    )
    return response


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readiness(request: Request) -> dict[str, str]:
    if request.app.state.model is None:
        raise HTTPException(status_code=503, detail="Model artifact is unavailable or invalid")
    return {"status": "ready"}


@app.get("/v1/model")
def model_info(request: Request) -> dict[str, Any]:
    metadata = request.app.state.metadata
    if metadata is None:
        raise HTTPException(status_code=503, detail="Model artifact is unavailable or invalid")
    return {
        key: metadata[key]
        for key in (
            "model_version",
            "schema_version",
            "model",
            "trained_at",
            "training_years",
            "holdout_year",
            "decision",
            "limitations",
        )
    }


@app.post("/v1/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:  # type: ignore[valid-type]
    if request.app.state.model is None:
        raise HTTPException(status_code=503, detail="Model artifact is unavailable or invalid")
    prediction = predict_one(
        request.app.state.model,
        request.app.state.metadata,
        cast(BaseModel, payload).model_dump(),
    )
    PREDICTIONS.observe(prediction.price)
    return PredictionResponse(
        predicted_price_2010_usd=round(prediction.price, 2),
        interval_90_low=round(prediction.interval_low, 2),
        interval_90_high=round(prediction.interval_high, 2),
        inflation_adjusted_reference_usd=round(adjust_2010_dollars(prediction.price), 2),
        inflation_adjusted_interval_90_low=round(adjust_2010_dollars(prediction.interval_low), 2),
        inflation_adjusted_interval_90_high=round(adjust_2010_dollars(prediction.interval_high), 2),
        inflation_reference_year=CPI_U_REFERENCE.reference_year,
        inflation_factor=round(CPI_U_REFERENCE.factor, 6),
        inflation_note=CPI_U_REFERENCE.note,
        model_version=request.app.state.metadata["model_version"],
        warnings=[*prediction.warnings, CPI_U_REFERENCE.note],
    )


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
