"""Model training module for the churn prediction pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

try:
    import mlflow
    import mlflow.sklearn
    from mlflow.models.signature import ModelSignature
    from mlflow.types.schema import ColSpec, Schema
except ImportError:  # pragma: no cover - used only when optional deps are absent.
    mlflow = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "churn_model.pkl"

LOG_DIR.mkdir(parents=True, exist_ok=True)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(LOG_DIR / "training.log")
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)
LOGGER.propagate = False

FEATURE_COLUMNS = [
    "age",
    "tenure",
    "monthly_charges",
    "num_products",
    "support_calls",
    "charge_per_tenure",
    "support_intensity",
    "is_high_value",
]
TARGET_COLUMN = "churn"


def load_data(data_path: str | Path = PROCESSED_DATA_PATH) -> tuple:
    """Load processed data and create a train/test split."""
    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Processed data not found at {data_file}. Run preprocessing first.")

    df = pd.read_csv(data_file)
    LOGGER.info("Loaded %s records from %s", len(df), data_file)

    missing_columns = [column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in processed data: {missing_columns}")

    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    LOGGER.info("Train set: %s rows, test set: %s rows", len(x_train), len(x_test))
    return x_train, x_test, y_train, y_test


def train_model(x_train, y_train, c_value: float = 1.0, max_iter: int = 1000) -> LogisticRegression:
    """Train the churn model."""
    model = LogisticRegression(C=c_value, max_iter=max_iter, random_state=42)
    model.fit(x_train, y_train)
    LOGGER.info("Model trained: C=%s max_iter=%s", c_value, max_iter)
    return model


def evaluate_model(model, x_test, y_test) -> dict:
    """Evaluate the model and return classification metrics."""
    y_pred = model.predict(x_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    for metric_name, value in metrics.items():
        LOGGER.info("%s: %.4f", metric_name, value)
    return metrics


def _configure_mlflow() -> None:
    if mlflow is None:
        LOGGER.warning("MLflow is not installed; skipping MLflow tracking for this run.")
        return

    mlflow.set_tracking_uri((PROJECT_ROOT / "mlflow").as_uri())
    mlflow.set_experiment("churn_prediction")


def _model_signature() -> ModelSignature:
    input_schema = Schema([ColSpec("double", column) for column in FEATURE_COLUMNS])
    output_schema = Schema([ColSpec("long")])
    return ModelSignature(inputs=input_schema, outputs=output_schema)


def train_with_mlflow(
    data_path: str | Path = PROCESSED_DATA_PATH,
    model_path: str | Path = MODEL_PATH,
    c_value: float = 1.0,
    max_iter: int = 1000,
):
    """Train, evaluate, save, and optionally log the model with MLflow."""
    _configure_mlflow()

    if mlflow is None:
        x_train, x_test, y_train, y_test = load_data(data_path)
        model = train_model(x_train, y_train, c_value=c_value, max_iter=max_iter)
        metrics = evaluate_model(model, x_test, y_test)
        output_path = Path(model_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_path)
        LOGGER.info("Model saved to %s", output_path)
        return model, metrics

    with mlflow.start_run():
        mlflow.log_param("C", c_value)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("features", len(FEATURE_COLUMNS))

        x_train, x_test, y_train, y_test = load_data(data_path)
        model = train_model(x_train, y_train, c_value=c_value, max_iter=max_iter)
        metrics = evaluate_model(model, x_test, y_test)

        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        output_path = Path(model_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_path)
        LOGGER.info("Model saved to %s", output_path)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name="churn_model",
            signature=_model_signature(),
            input_example=x_train.head(5).astype("float64"),
        )
        mlflow.log_param("data_path", str(data_path))
        mlflow.log_param("model_path", str(model_path))
        LOGGER.info("Model logged to MLflow")
        return model, metrics


def train_and_save_model() -> Path:
    """Compatibility entrypoint for older scripts."""
    train_with_mlflow()
    return MODEL_PATH


if __name__ == "__main__":
    try:
        _, run_metrics = train_with_mlflow()
        print(f"Saved model: {MODEL_PATH}")
        print(f"Metrics: {run_metrics}")
    except Exception:
        print("Training failed. Check logs for details.")
        sys.exit(1)
