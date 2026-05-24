"""Compatibility wrapper for the new full pipeline script."""

import argparse
import sys
import time

from scripts.full_pipeline import run_full_pipeline


def retrain(rows: int, seed: int | None) -> None:
    final_seed = int(time.time()) % 100000 if seed is None else seed
    result = run_full_pipeline(n_samples=rows, seed=final_seed)
    if result["status"] != "success":
        raise RuntimeError(result["error"])
    print("Retraining complete. A new model version has been saved.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full churn MLOps pipeline.")
    parser.add_argument("--rows", type=int, default=1400, help="Number of synthetic rows for retraining.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed.")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()
        retrain(rows=args.rows, seed=args.seed)
    except Exception:
        print("Retraining failed. Check logs for details.")
        sys.exit(1)
