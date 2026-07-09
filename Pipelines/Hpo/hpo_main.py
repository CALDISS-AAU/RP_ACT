"""Main script for the Hpo pipeline."""
import argparse
# Internal imports
from Shared_Functions.modelling_functions import (
    build_sentence_transformer,
    prepare_tsdae_data
)
from Functions.hpo_func import (
    _hpo_loss_init,
    _hpo_compute_objective,
    _run_tsdae_hpo
)

# Directories
INPUT_DIR_DATA = "/work/RP_act/ACT/Data/clean_combine/message_data.parquet"
OUTPUT_DIR_LOG = "/work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/Hpo/Logs/hpo.log"
OUTPUT_DIR = '/work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/Hpo/output'

def main():
    parser = argparse.ArgumentParser(description="Perform HPO on TSDAE model")
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to input .parquet file"
    )
    parser.add_argument(
        '--output-dir-logs',
        default=str(REPO_ROOT / "Pipelines" / 'Hpo' / "Logs"),
        help="Directory for storing HPO logs",
    )

    args = parser.parse_args()
    _run_tsdae_hpo(input_dir=args.data_path, output_dir=OUTPUT_DIR)

if __name__ == "__main__":
    main()