"""Main script for the Conversation_Extraction pipeline."""

"""Run the ACT clean-and-combine pipeline. 
   The script inspects raw conversation data, 
       cleans and combines the raw CSV files, 
       adds derived variables, exports the final dataset, 
       and rebuilds the full pipeline log. 

   Run from the ACT folder with: 
       python -m Pipelines.Conversation_Extraction.conversation_extraction_main 
"""
## IMPORTS ##
# Internal
from Shared_Functions.logger_functionality import *
from .Functions.data_investigation import investigate_data_stats

## _______ ##

## STATIC VARIABLES ##
# Directories - input
INPUT_DIR_CONVERSATIONS = "/work/ACT/Data/clean_combine/conversation_data.parquet" 

# Directories - internal output
OUTPUT_DIR_DATA_INVESTIGATION_PNGS = "Pipelines/Conversation_Extraction/Data/pngs" 

# Directories - global output
OUTPUT_DIR_FINAL_DATA_INVESTIGATION = "./Data/conversation_extraction/data_investiogation.docx"

# Directories - logs
OUTPUT_DIR_LOG_FULL_PIPELINE = "./Pipelines/Conversation_Extraction/Logs/full_conversation_extraction_pipeline.log"
OUTPUT_DIR_LOG_DATA_INVESTIGATION = "./Pipelines/Conversation_Extraction/Logs/data_investigation.log"

## _______________________ ##

## HELPER FUNCTIONS ##
## ________________ ##

## MAIN FUNCTION ##
def main() -> None: 
    """Run all conversation-extraction pipeline steps."""
    investigate_data_stats(
    INPUT_DIR_CONVERSATIONS,
    OUTPUT_DIR_FINAL_DATA_INVESTIGATION,
    output_dir_pngs=OUTPUT_DIR_DATA_INVESTIGATION_PNGS,
    logger=setup_logger(
        output_dir_log=OUTPUT_DIR_LOG_DATA_INVESTIGATION,
        logger_name="conversation_extraction.data_inspection",
    ),
    bottom_percent=25.0,
    wordcount_interval_size=200,
    wordcount_stop_at=2000,
)

## _____________ ##

## CALL OF MAIN FUNCTION ##
if __name__ == "__main__":
    main()