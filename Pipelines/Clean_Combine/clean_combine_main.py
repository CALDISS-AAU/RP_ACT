""" Main script for the Clean_Combine pipeline.
    Run this script by going to the ACT folder and then run:
        python -m Pipelines.Clean_Combine.clean_combine_main
"""
## IMPORTS ##
# Internal
from .Functions.data_inspection import inspect_raw_data 
from .Functions.raw_data_clean_and_combine import clean_and_combine_data
## _______ ##

## STATIC VARIABLES ##
# Directories
INPUT_DIR_CONVERSATIONS = "./Data/raw/samtaler"

OUTPUT_DIR_COMBINED_DATA = "Pipelines/Clean_Combine/Data/combined_data.parquet"

OUTPUT_DIR_LOG_INSPECTION = "./Pipelines/Clean_Combine/Logs/inspection_of_raw_data.log"
OUTPUT_DIR_LOG_CLEAN_AND_COMBINE = "./Pipelines/Clean_Combine/Logs/clean_and_combine_raw_data.log"
OUTPUT_DIR_LOG_CLEAN_AND_COMBINE_EVALUATION = "./Pipelines/Clean_Combine/Logs/clean_and_combine_raw_data_evaluation.log"

# Groupings
age_ranges = {
    "pre teen": [10, 11, 12],
    "early teen": [13, 14],
    "teen": [15, 16. 17]
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
    # clean_and_combine_data(INPUT_DIR_CONVERSATIONS, OUTPUT_DIR_COMBINED_DATA, OUTPUT_DIR_LOG_CLEAN_AND_COMBINE)


## _____________ ##

## CALL OF MAIN FUNCTION ##
if __name__ == "__main__":
    main()