"""Simple file-based model registry."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ModelRegistry:
    """Store model artifacts and metadata by model name and version."""

    def __init__(self, registry_path: str | Path = PROJECT_ROOT / "model_registry") -> None:
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

    def save_model(
        self,
        model,
        model_name: str,
        version: str,
        metrics: Dict,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Save a model artifact with metadata."""
        model_dir = self.registry_path / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "model.pkl"
        joblib.dump(model, model_path)

        metadata_file = model_dir / "metadata.json"
        registry_metadata = {
            "model_name": model_name,
            "version": version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "custom_metadata": metadata or {},
        }
        metadata_file.write_text(json.dumps(registry_metadata, indent=2), encoding="utf-8")

        print(f"Model saved: {model_name}@{version}")
        return str(model_path)

    def load_model(self, model_name: str, version: str):
        """Load a model by name and version."""
        model_path = self.registry_path / model_name / version / "model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_name}@{version}")

        model = joblib.load(model_path)
        print(f"Model loaded: {model_name}@{version}")
        return model

    def get_metadata(self, model_name: str, version: str) -> Dict:
        """Return model metadata."""
        metadata_path = self.registry_path / model_name / version / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found for {model_name}@{version}")

        return json.loads(metadata_path.read_text(encoding="utf-8"))

    def list_versions(self, model_name: str) -> list[str]:
        """List all versions for a registered model."""
        model_dir = self.registry_path / model_name
        if not model_dir.exists():
            return []

        return sorted(path.name for path in model_dir.iterdir() if path.is_dir())


if __name__ == "__main__":
    from training.train import train_with_mlflow

    trained_model, metrics = train_with_mlflow()
    registry = ModelRegistry()
    registry.save_model(
        trained_model,
        model_name="churn_model",
        version="v1.0",
        metrics=metrics,
        metadata={"environment": "development"},
    )
    print(f"Available versions: {registry.list_versions('churn_model')}")
