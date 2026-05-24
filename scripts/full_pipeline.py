"""Full MLOps pipeline: ingest -> preprocess -> train -> registry."""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.ingest import ingest
from data_pipeline.preprocess import preprocess
from model.model_registry import ModelRegistry
from training.train import train_with_mlflow


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
LOGGER = logging.getLogger(__name__)


def run_full_pipeline(n_samples: int = 1000, seed: int = 42) -> dict:
    """Execute the complete ML workflow."""
    LOGGER.info("=" * 60)
    LOGGER.info("Starting Full MLOps Pipeline")
    LOGGER.info("=" * 60)

    try:
        LOGGER.info("[1/4] Running data ingestion")
        raw_data_path = ingest(n_samples=n_samples, seed=seed)

        LOGGER.info("[2/4] Running data preprocessing")
        processed_data_path = preprocess(input_path=raw_data_path)

        LOGGER.info("[3/4] Running model training")
        model, metrics = train_with_mlflow(data_path=processed_data_path)

        LOGGER.info("[4/4] Saving to model registry")
        registry = ModelRegistry()
        version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_path = registry.save_model(
            model,
            model_name="churn_model",
            version=version,
            metrics=metrics,
            metadata={
                "pipeline_run_date": datetime.now(timezone.utc).isoformat(),
                "data_source": raw_data_path,
            },
        )

        result = {
            "status": "success",
            "model_path": model_path,
            "metrics": metrics,
            "version": version,
        }
        LOGGER.info("Full pipeline completed successfully: %s", result)
        return result

    except Exception as exc:
        LOGGER.exception("Pipeline failed")
        return {"status": "failed", "error": str(exc)}


if __name__ == "__main__":
    pipeline_result = run_full_pipeline()
    sys.exit(0 if pipeline_result["status"] == "success" else 1)
