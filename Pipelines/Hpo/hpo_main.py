"""Main script for the Hpo pipeline."""
import argparse
import os
from pathlib import Path

from multiprocessing import get_context
# Internal imports
from Shared_Functions.modelling_functions import (
    build_sentence_transformer,
    # prepare_tsdae_data
)
from Pipelines.Data_Prep.Functions.data_func import (
    _prepare_tsdae_data
)

from Functions.hpo_func import (
    _hpo_loss_init,
    _hpo_compute_objective,
    _run_tsdae_hpo
)

REPO_ROOT = Path(os.environ.get("ACT_REPO_ROOT", ".")).resolve()


def main():
    parser = argparse.ArgumentParser(description="Perform HPO on TSDAE model")
    # parser.add_argument(
    #     "--prepare-data",
    #     action="store_true",
    #     help="Create and save HF train/validation/test dataset before HPO"
    # )

    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(REPO_ROOT / "Pipelines" / "Hpo" / "prepared_dataset"),
        help="Directory where the prepared HF DatasetDict is saved.",
    )

    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to input HF dataset file"
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
        default=4,
        help=(
            "Number of trials to run in this worker"
        ),
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="Max sequence length of texts - should be 128 for messages and 1024 for conversations."
    )

    parser.add_argument(
        "--study_name",
        type=str,
        default="Study_name",
        required=True,
        help="Give a unique study name",
    )

    args = parser.parse_args()

    # if args.prepare_data:
    #     _prepare_tsdae_data(
    #         data_path=args.data_path,
    #         base_dir=args.dataset_dir,
    #     )
    
    _run_tsdae_hpo(
        input_dir=args.dataset_dir,
        output_dir=args.output_trial_dir,
        log_dir=args.output_dir_logs,
        n_trials=args.n_trials,
        max_seq_length=args.max_seq_length,
        study_name=args.study_name
    )


if __name__ == "__main__":
    main()
