"""Main script for the Hpo pipeline."""
import argparse
import os
from pathlib import Path

from Functions.hpo_func import _run_tsdae_hpo

# Internal imports

REPO_ROOT = Path(os.environ.get("ACT_REPO_ROOT", ".")).resolve()


def main():
    parser = argparse.ArgumentParser(description="Perform HPO on TSDAE model")
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to input .parquet file"
    )
    parser.add_argument(
        "--output-trial-dir",
        default=(REPO_ROOT / "Pipelines" / "Hpo" / "output")
    )
    parser.add_argument(
        '--output-dir-logs',
        default=str(REPO_ROOT / "Pipelines" / 'Hpo' / "Logs"),
        help="Directory for storing HPO trial outputs",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=5,
        help=(
            "Number of trials to run in this worker"
        ),
    )

    args = parser.parse_args()
    
    
    _run_tsdae_hpo(
        input_dir=args.data_path,
        output_dir=args.output_trial_dir,
        log_dir=args.output_dir_logs,
        n_trials=args.n_trials,
        )


if __name__ == "__main__":
    main()
