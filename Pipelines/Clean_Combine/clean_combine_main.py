""" Main script for the Clean_Combine pipeline.
    Run this script by going to the ACT folder and then run:
        python -m Pipelines.Clean_Combine.clean_combine_main
"""
## IMPORTS ##
# Internal
from .Functions.data_inspection import inspect_raw_data 
from .Functions.raw_data_clean_and_combine import clean_and_combine_data
from .Functions.add_variables import add_variables
## _______ ##

## STATIC VARIABLES ##
# Directories
INPUT_DIR_CONVERSATIONS = "/work/ACT/Data/raw/samtaler" # "./Data/raw/samtaler"
INPUT_DIR_CONTACT_REASONS = "/work/ACT/Data/raw/emner" # "./Data/raw/emner"

OUTPUT_DIR_COMBINED_DATA = "Pipelines/Clean_Combine/Data/cleaned_data.parquet"
OUTPUT_DIR_FINAL_DATA = "./Data/clean_combined_data.parquet"

OUTPUT_DIR_LOG_INSPECTION = "./Pipelines/Clean_Combine/Logs/inspection_of_raw_data.log"
OUTPUT_DIR_LOG_CLEAN_AND_COMBINE = "./Pipelines/Clean_Combine/Logs/clean_and_combine_raw_data.log"
OUTPUT_DIR_LOG_CLEAN_AND_COMBINE_EVALUATION = "./Pipelines/Clean_Combine/Logs/clean_and_combine_raw_data_evaluation.log"
OUTPUT_DIR_LOG_VARIABLE_ADDITION = "./Pipelines/Clean_Combine/Logs/variable_addition.log"

# Groupings
age_ranges = {
    "pre teen": [10, 11, 12],
    "early teen": [13, 14],
    "teen": [15, 16, 17]
}

time_of_day = {
    "night": range(0, 6),
    "morning": range(6, 12),
    "midday": range(12, 13),
    "afternoon": range(13, 18),
    "evening": range(18, 24)
}

season = {
    "winter": [12, 1, 2],
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "autumn": [9, 10, 11]
}
## _______________________ ##

## HELPER FUNCTIONS ##
## ________________ ##

## MAIN FUNCTION ##
def main():
    # inspect_raw_data(INPUT_DIR_CONVERSATIONS, OUTPUT_DIR_LOG_INSPECTION)
    # clean_and_combine_data(INPUT_DIR_CONVERSATIONS, OUTPUT_DIR_COMBINED_DATA, OUTPUT_DIR_LOG_INSPECTION)
    add_variables(
        input_dir_data=OUTPUT_DIR_COMBINED_DATA, 
        input_dir_contact_reason=INPUT_DIR_CONTACT_REASONS, 
        output_dir_data=OUTPUT_DIR_FINAL_DATA, 
        output_dir_log=OUTPUT_DIR_LOG_VARIABLE_ADDITION,
        age_range_dict=age_ranges,
        time_of_day_dict=time_of_day,
        season_dict=season
    )

## _____________ ##

## CALL OF MAIN FUNCTION ##
if __name__ == "__main__":
    main()