"""Main script for the Tsdae_Model pipeline."""
# Internal imports 
from Functions.train_func import *

# Directories
INPUT_DIR_DATA = "/work/RP_act/ACT/Data/clean_combine/message_data.parquet"
OUTPUT_DIR_LOG_TRAIN = "/work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/Tsdae_Model/Logs/trainer.log"
OUTPUT_DIR_MODEL = '/work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/Tsdae_Model/ACT_messagelvl_embed'

def main():
    train_message_model(input_dir=INPUT_DIR_DATA, output_dir=OUTPUT_DIR_MODEL, log=OUTPUT_DIR_LOG_TRAIN)


if __name__ == "__main__":
    main()

