"""
Data preprocessing module.

Validates raw churn data, cleans missing values, engineers features, and writes
monitoring statistics.
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "raw_data.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"
STATS_PATH = PROJECT_ROOT / "data" / "processed" / "stats.json"

FEATURE_COLUMNS = ["age", "tenure", "monthly_charges", "num_products", "support_calls"]
ENGINEERED_FEATURE_COLUMNS = ["charge_per_tenure", "support_intensity", "is_high_value"]
TARGET_COLUMN = "churn"
ALL_FEATURE_COLUMNS = FEATURE_COLUMNS + ENGINEERED_FEATURE_COLUMNS

LOG_DIR.mkdir(parents=True, exist_ok=True)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(LOG_DIR / "preprocess.log")
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)
LOGGER.propagate = False


def validate_schema(df: pd.DataFrame) -> None:
    """Validate that the raw dataset contains all required columns."""
    expected = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Schema validation failed. Missing columns: {sorted(missing)}")
    LOGGER.info("Schema validation passed")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Handle duplicates, numeric coercion, and missing values."""
    processed = df.copy()
    original_len = len(processed)
    processed = processed.drop_duplicates()

    for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
        processed[column] = pd.to_numeric(processed[column], errors="coerce")

    for column in FEATURE_COLUMNS:
        if processed[column].isnull().sum() > 0:
            median_value = processed[column].median()
            processed[column] = processed[column].fillna(median_value)
            LOGGER.info("Filled %s nulls with median=%.4f", column, median_value)

    processed = processed.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    for column in ["age", "tenure", "num_products", "support_calls", TARGET_COLUMN]:
        processed[column] = processed[column].round().astype(int)

    LOGGER.info(
        "Cleaning complete: %s -> %s rows (removed %s rows)",
        original_len,
        len(processed),
        original_len - len(processed),
    )
    return processed


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model features used by both training and inference."""
    processed = df.copy()
    high_value_threshold = processed["monthly_charges"].median()

    processed["charge_per_tenure"] = processed["monthly_charges"] / (processed["tenure"] + 1)
    processed["support_intensity"] = processed["support_calls"] / (processed["tenure"] + 1)
    processed["is_high_value"] = (processed["monthly_charges"] > high_value_threshold).astype(int)

    LOGGER.info("Feature engineering complete: created %s", ENGINEERED_FEATURE_COLUMNS)
    return processed


def compute_statistics(df: pd.DataFrame) -> dict:
    """Compute feature statistics for validation and simple monitoring."""
    stats = {}
    for column in ALL_FEATURE_COLUMNS:
        stats[column] = {
            "mean": round(float(df[column].mean()), 4),
            "std": round(float(df[column].std()), 4),
            "min": round(float(df[column].min()), 4),
            "max": round(float(df[column].max()), 4),
            "median": round(float(df[column].median()), 4),
            "nulls": int(df[column].isnull().sum()),
        }

    stats["_metadata"] = {
        "n_rows": int(len(df)),
        "n_features": len(ALL_FEATURE_COLUMNS),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "churn_rate": round(float(df[TARGET_COLUMN].mean()), 4),
    }
    return stats


def preprocess(
    input_path: str | Path = RAW_DATA_PATH,
    output_path: str | Path = PROCESSED_DATA_PATH,
    stats_path: str | Path = STATS_PATH,
) -> str:
    """Run the complete preprocessing pipeline."""
    input_file = Path(input_path)
    output_file = Path(output_path)
    stats_file = Path(stats_path)

    LOGGER.info("Starting preprocessing: input=%s output=%s", input_file, output_file)
    df = pd.read_csv(input_file)
    LOGGER.info("Loaded %s records from %s", len(df), input_file)

    validate_schema(df)
    processed = engineer_features(clean(df))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_file, index=False)
    LOGGER.info("Saved processed data to %s", output_file)

    stats_file.parent.mkdir(parents=True, exist_ok=True)
    stats_file.write_text(json.dumps(compute_statistics(processed), indent=2), encoding="utf-8")
    LOGGER.info("Saved data statistics to %s", stats_file)
    return str(output_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess raw churn data.")
    parser.add_argument("--input", type=Path, default=RAW_DATA_PATH, help="Raw input CSV path.")
    parser.add_argument("--output", type=Path, default=PROCESSED_DATA_PATH, help="Processed output CSV path.")
    parser.add_argument("--stats", type=Path, default=STATS_PATH, help="Statistics JSON path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess(input_path=args.input, output_path=args.output, stats_path=args.stats)
