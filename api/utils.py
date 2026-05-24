"""Shared inference utilities for the churn prediction API."""

import json
from pathlib import Path

import pandas as pd

from api.schemas import ChurnPredictionInput


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATS_PATH = PROJECT_ROOT / "data" / "processed" / "stats.json"
DEFAULT_HIGH_VALUE_THRESHOLD = 65.0

FEATURE_ORDER = [
    "age",
    "tenure",
    "monthly_charges",
    "num_products",
    "support_calls",
    "charge_per_tenure",
    "support_intensity",
    "is_high_value",
]


def load_high_value_threshold(stats_path: str | Path = STATS_PATH) -> float:
    """Use training data statistics when available to avoid training-serving skew."""
    stats_file = Path(stats_path)
    if not stats_file.exists():
        return DEFAULT_HIGH_VALUE_THRESHOLD

    try:
        stats = json.loads(stats_file.read_text(encoding="utf-8"))
        return float(stats["monthly_charges"]["median"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return DEFAULT_HIGH_VALUE_THRESHOLD


def build_feature_frame(input_data: ChurnPredictionInput, high_value_threshold: float) -> pd.DataFrame:
    """Create the model feature frame from a validated request payload."""
    charge_per_tenure = input_data.monthly_charges / (input_data.tenure + 1)
    support_intensity = input_data.support_calls / (input_data.tenure + 1)
    is_high_value = 1 if input_data.monthly_charges > high_value_threshold else 0

    features = {
        "age": input_data.age,
        "tenure": input_data.tenure,
        "monthly_charges": input_data.monthly_charges,
        "num_products": input_data.num_products,
        "support_calls": input_data.support_calls,
        "charge_per_tenure": charge_per_tenure,
        "support_intensity": support_intensity,
        "is_high_value": is_high_value,
    }
    return pd.DataFrame([features], columns=FEATURE_ORDER)
