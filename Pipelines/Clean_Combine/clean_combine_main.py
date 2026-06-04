"""Main script for the Clean_Combine pipeline."""
## IMPORTS ##
# External
# Internal
from .Functions.inspection_of_raw_data import inspect_data 
## _______ ##

## VARIABLES TO BE CHANGED ##
INPUT_DIR_CONVERSATIONS = "./Data/raw/samtaler"
OUTPUT_DIR_LOG_INSPECTION = "./Pipelines/Clean_Combine/Logs/inspection_of_raw_data.log"
## _______________________ ##

## HELPER FUNCTIONS ##
## ________________ ##

## MAIN FUNCTION ##
def main():
    inspect_data(INPUT_DIR_CONVERSATIONS, OUTPUT_DIR_LOG_INSPECTION)
## _____________ ##

## CALL OF MAIN FUNCTION ##
if __name__ == "__main__":
    main()