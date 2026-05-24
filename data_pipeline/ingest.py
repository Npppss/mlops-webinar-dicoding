"""
Data ingestion module for the churn prediction demo.

This simulates pulling customer data from an external system and writes the raw
dataset to disk.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "raw_data.csv"

LOG_DIR.mkdir(parents=True, exist_ok=True)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(LOG_DIR / "ingest.log")
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)
LOGGER.propagate = False


def generate_synthetic_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic customer churn dataset."""
    rng = np.random.default_rng(seed)

    data = {
        "age": rng.integers(18, 70, n_samples),
        "tenure": rng.integers(1, 72, n_samples),
        "monthly_charges": np.round(rng.uniform(20, 120, n_samples), 2),
        "num_products": rng.integers(1, 5, n_samples),
        "support_calls": rng.integers(0, 10, n_samples),
    }
    df = pd.DataFrame(data)

    null_mask = rng.random(n_samples) < 0.03
    df.loc[null_mask, "monthly_charges"] = np.nan

    monthly_component = df["monthly_charges"].fillna(df["monthly_charges"].median()) / 120
    churn_prob = (
        0.3 * (df["support_calls"] / 10)
        + 0.3 * monthly_component
        + 0.2 * (1 - df["tenure"] / 72)
        + 0.2 * rng.random(n_samples)
    )
    df["churn"] = (churn_prob > 0.5).astype(int)
    return df


def ingest(
    output_path: str | Path = RAW_DATA_PATH,
    n_samples: int = 1000,
    seed: int = 42,
) -> str:
    """Generate and save raw data."""
    output = Path(output_path)
    LOGGER.info("Starting data ingestion: target=%s n_samples=%s seed=%s", output, n_samples, seed)

    output.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_data(n_samples=n_samples, seed=seed)
    df.to_csv(output, index=False)

    LOGGER.info("Ingested %s records", len(df))
    LOGGER.info("Columns: %s", list(df.columns))
    LOGGER.info("Shape: %s", df.shape)
    LOGGER.info("Null values:\n%s", df.isnull().sum())
    return str(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic raw churn data.")
    parser.add_argument("--rows", type=int, default=1000, help="Number of synthetic records.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output", type=Path, default=RAW_DATA_PATH, help="Raw output CSV path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ingest(output_path=args.output, n_samples=args.rows, seed=args.seed)
