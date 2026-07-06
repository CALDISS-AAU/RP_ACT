"""Main script for the Hpo pipeline."""
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
    _run_tsdae_hpo(input_dir=INPUT_DIR_DATA, output_dir=OUTPUT_DIR, log=OUTPUT_DIR_LOG)

if __name__ == "__main__":
    main()