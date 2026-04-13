"""Run the refactored coursework pipeline end to end."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from interpreterule.pipeline import run_full_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the interpretable ML coursework refactor.")
    parser.add_argument(
        "--data",
        type=str,
        default=str(ROOT / "data" / "raw" / "machlearnia.csv"),
        help="Path to the input CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "outputs"),
        help="Directory where figures and tables will be saved.",
    )
    parser.add_argument(
        "--skip-extensions",
        action="store_true",
        help="Skip the low-priority tournament simulation extension.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_full_pipeline(
        dataset_path=args.data,
        output_dir=args.output_dir,
        include_extensions=not args.skip_extensions,
    )
    print("Saved predictions to:", artifacts.predictions_path)
    print("Saved metrics to:", artifacts.metrics_path)
    print("Saved rank rules to:", artifacts.rule_table_path)
    print("Saved formula coefficients to:", artifacts.formula_table_path)


if __name__ == "__main__":
    main()
