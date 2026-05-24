"""FastAPI application for churn model inference."""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import ChurnPredictionInput, ChurnPredictionOutput, HealthCheck
from api.utils import build_feature_frame, load_high_value_threshold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
MODEL_PATH = PROJECT_ROOT / "model" / "churn_model.pkl"

LOG_DIR.mkdir(parents=True, exist_ok=True)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(LOG_DIR / "api.log")
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)
LOGGER.propagate = False

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using a trained ML model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
MODEL_VERSION = "v1.0"
HIGH_VALUE_THRESHOLD = 65.0


def load_model(model_path: str | Path = MODEL_PATH):
    """Load the trained model from disk."""
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found at {model_file}")
    return joblib.load(model_file)


@app.on_event("startup")
async def startup_event() -> None:
    """Load model and feature statistics on startup."""
    global MODEL, HIGH_VALUE_THRESHOLD

    try:
        MODEL = load_model()
        HIGH_VALUE_THRESHOLD = load_high_value_threshold()
        LOGGER.info("Model loaded from %s", MODEL_PATH)
        LOGGER.info("High-value threshold set to %.4f", HIGH_VALUE_THRESHOLD)
    except Exception as exc:
        MODEL = None
        LOGGER.error("API startup could not load model: %s", exc)


@app.get("/", tags=["Info"])
def root() -> dict:
    """Root endpoint."""
    return {
        "service": "Customer Churn Prediction API",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health", response_model=HealthCheck, tags=["Health"])
def health_check() -> HealthCheck:
    """Return service and model status."""
    model_loaded = MODEL is not None
    return HealthCheck(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        model_version=MODEL_VERSION,
    )


@app.post("/predict", response_model=ChurnPredictionOutput, tags=["Prediction"])
def predict(input_data: ChurnPredictionInput) -> ChurnPredictionOutput:
    """Predict customer churn for a single record."""
    if MODEL is None:
        LOGGER.error("Prediction requested before model was loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        feature_frame = build_feature_frame(input_data, HIGH_VALUE_THRESHOLD)
        prediction = int(MODEL.predict(feature_frame)[0])
        probability = float(MODEL.predict_proba(feature_frame)[0][1])

        LOGGER.info(
            "Prediction: input=%s prediction=%s probability=%.4f",
            input_data.model_dump(),
            prediction,
            probability,
        )

        return ChurnPredictionOutput(
            prediction=prediction,
            probability_churn=round(probability, 4),
            model_version=MODEL_VERSION,
        )
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.post("/predict-batch", tags=["Prediction"])
def predict_batch(inputs: List[ChurnPredictionInput]) -> dict:
    """Predict customer churn for multiple records."""
    results = [predict(input_data) for input_data in inputs]
    LOGGER.info("Batch prediction processed %s records", len(results))
    return {"predictions": results, "count": len(results)}


@app.get("/metrics", tags=["Monitoring"])
def get_metrics() -> dict:
    """Return basic operational metrics."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": MODEL_VERSION,
        "model_loaded": MODEL is not None,
        "api_status": "operational" if MODEL is not None else "degraded",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
