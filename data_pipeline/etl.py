"""Compatibility ETL wrapper for the new ingest + preprocess pipeline."""

import argparse
from pathlib import Path

from data_pipeline.ingest import ingest
from data_pipeline.preprocess import PROCESSED_DATA_PATH, preprocess


def run_etl(n_rows: int, seed: int, output_path: Path = PROCESSED_DATA_PATH) -> Path:
    raw_path = ingest(n_samples=n_rows, seed=seed)
    processed_path = preprocess(input_path=raw_path, output_path=output_path)
    return Path(processed_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run churn ingest and preprocessing pipeline.")
    parser.add_argument("--rows", type=int, default=1000, help="Number of synthetic rows to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_etl(n_rows=args.rows, seed=args.seed)
