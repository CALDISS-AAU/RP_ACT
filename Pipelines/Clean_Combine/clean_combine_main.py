"""Run the ACT clean-and-combine pipeline.
   The script inspects raw conversation data,
       cleans and combines the raw CSV files,
       generates working datasets,
       adds derived variables, exports the final datasets,
       and rebuilds the full pipeline log.

   Run from the ACT folder with:
       uv run python -m Pipelines.Clean_Combine.clean_combine_main
"""

## IMPORTS ##
# Internal
from Shared_Functions.logger_functionality import *
from .Functions.data_inspection import inspect_raw_data 
from .Functions.raw_data_clean_and_combine import clean_and_combine_data
from .Functions.generate_working_datasets import generate_working_datasets
# from .Functions.add_variables import add_variables
## _______ ##

## STATIC VARIABLES ##
# Directories - input
INPUT_DIR_CONVERSATIONS = "/work/RP_act/ACT/Data/raw/samtaler"
INPUT_DIR_CONTACT_REASONS = "/work/RP_act/ACT/Data/raw/emner"

# Directories - internal output
OUTPUT_DIR_MESSAGE_DATA = "Pipelines/Clean_Combine/Data/message_data.parquet" 

# Directories - global output
OUTPUT_DIR_FOLDER = "./Data/Clean_Combine/"

# Directories - logs
OUTPUT_DIR_LOG_FULL_PIPELINE = "./Pipelines/Clean_Combine/Logs/full_clean_combine_pipeline.log"
OUTPUT_DIR_LOG_INSPECTION = "./Pipelines/Clean_Combine/Logs/inspection_of_raw_data.log"
OUTPUT_DIR_LOG_CLEAN_AND_COMBINE = "./Pipelines/Clean_Combine/Logs/clean_and_combine_raw_data.log"
OUTPUT_DIR_LOG_CLEAN_AND_COMBINE_EVALUATION = "./Pipelines/Clean_Combine/Logs/clean_and_combine_raw_data_evaluation.log"
OUTPUT_DIR_LOG_WORKING_DATASETS = "./Pipelines/Clean_Combine/Logs/generate_working_datasets.log"

# Groupings
AGE_RANGES = {
    "pre teen": [10, 11, 12],
    "early teen": [13, 14],
    "teen": [15, 16, 17]
}

TIME_OF_DAY = {
    "night": range(0, 6),
    "morning": range(6, 12),
    "midday": range(12, 13),
    "afternoon": range(13, 18),
    "evening": range(18, 24)
}

SEASON = {
    "winter": [12, 1, 2],
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "autumn": [9, 10, 11]
}

# Other
MESSAGE_SEPARATOR = " "
## _______________________ ##

## HELPER FUNCTIONS ##
## ________________ ##

## MAIN FUNCTION ##
def main() -> None: 
    """Run the full ACT clean-and-combine pipeline.
       Executes raw data inspection, cleaning and combination,
           working-dataset generation, variable addition for all output
           datasets, and rebuilding of the combined pipeline log.
    """
    inspect_raw_data(
        input_dir=INPUT_DIR_CONVERSATIONS,
        logger=setup_logger(
            output_dir_log=OUTPUT_DIR_LOG_INSPECTION,
            logger_name="clean_combine.inspection",
        ),
    )

    clean_and_combine_data(
        input_dir=INPUT_DIR_CONVERSATIONS,
        output_dir_combined_data=OUTPUT_DIR_MESSAGE_DATA,
        cleaning_logger=setup_logger(
            OUTPUT_DIR_LOG_CLEAN_AND_COMBINE,
            "clean_combine.cleaning",
        ),
        evaluation_logger=setup_logger(
            OUTPUT_DIR_LOG_CLEAN_AND_COMBINE_EVALUATION,
            "clean_combine.evaluation",
        ),
    )

    generate_working_datasets(
        input_dir_data=OUTPUT_DIR_MESSAGE_DATA,
        input_dir_contact_reason=INPUT_DIR_CONTACT_REASONS,
        output_dir_folder=OUTPUT_DIR_FOLDER,
        logger=setup_logger(
            output_dir_log=OUTPUT_DIR_LOG_WORKING_DATASETS,
            logger_name="clean_combine.generate_working_datasets",
        ),
        separator=MESSAGE_SEPARATOR,
        age_range_dict=AGE_RANGES,
        time_of_day_dict=TIME_OF_DAY,
        season_dict=SEASON,
    )

    rebuild_pipeline_log(
        step_log_paths=[
            OUTPUT_DIR_LOG_INSPECTION,
            OUTPUT_DIR_LOG_CLEAN_AND_COMBINE,
            OUTPUT_DIR_LOG_WORKING_DATASETS,
        ],
        output_dir_log=OUTPUT_DIR_LOG_FULL_PIPELINE,
    )
## _____________ ##

## CALL OF MAIN FUNCTION ##
if __name__ == "__main__":
    main()