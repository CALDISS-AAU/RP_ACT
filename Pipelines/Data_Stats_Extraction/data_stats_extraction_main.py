"""Main script for the Conversation_Extraction pipeline."""

"""Run the ACT clean-and-combine pipeline. 
   The script inspects raw conversation data, 
       cleans and combines the raw CSV files, 
       adds derived variables, exports the final dataset, 
       and rebuilds the full pipeline log. 

   Run from the ACT folder with: 
       python -m Pipelines.Data_Stats_Extraction.data_stats_extraction_main 
"""
## IMPORTS ##
# Internal
from Shared_Functions.logger_functionality import *
from .Functions.data_investigation import *

## _______ ##

## STATIC VARIABLES ##
# Directories - input
INPUT_DIR_BASE = "/work/ACT/Data/clean_combine"
INPUT_DIR_MESSAGES = f"{INPUT_DIR_BASE}/message_data.parquet" 
INPUT_DIR_COLLAPSED_MESSAGES = f"{INPUT_DIR_BASE}/collapsed_messages_data.parquet" 
INPUT_DIR_CONVERSATIONS = f"{INPUT_DIR_BASE}/conversation_data.parquet" 

# Directories - global output
OUTPUT_DIR_DATA_STATS = "./Data/Data_Stats_Extraction/data_stats.jsonl" #"/work/ACT/Data/Data_Stats_Extraction/data_stats.jsonl"

# Directories - logs
OUTPUT_DIR_LOG_BASE = "./Pipelines/Data_Stats_Extraction/Logs"
OUTPUT_DIR_LOG_FULL_PIPELINE = f"{OUTPUT_DIR_LOG_BASE}/full_data_stats_extraction_pipeline.log"
OUTPUT_DIR_LOG_MESSAGES = f"{OUTPUT_DIR_LOG_BASE}/messages.log"
OUTPUT_DIR_LOG_COLLAPSED_MESSAGES = f"{OUTPUT_DIR_LOG_BASE}/collapsed_messages.log"
OUTPUT_DIR_LOG_CONVERSATIONS = f"{OUTPUT_DIR_LOG_BASE}/conversations.log"

# Other

## _______________________ ##

## HELPER FUNCTIONS ##
## ________________ ##

## MAIN FUNCTION ##
def main() -> None: 
    """Run all data-stats-extraction pipeline steps."""
    data_investigation_messages(
        INPUT_DIR_MESSAGES,
        OUTPUT_DIR_DATA_STATS,
        logger=setup_logger(
            OUTPUT_DIR_LOG_MESSAGES,
            logger_name="data_stats_extraction.message_data_investigation"
        )
    )

    # data_investigation_collapsed_messages(
    #     INPUT_DIR_COLLAPSED_MESSAGES,
    #     OUTPUT_DIR_DATA_STATS,
    #     logger=setup_logger(
    #         OUTPUT_DIR_LOG_COLLAPSED_MESSAGES,
    #         logger_name="data_stats_extraction.collapsed_message_data_investigation"
    #     )
    # )

    # data_investigation_conversations(
    #     INPUT_DIR_CONVERSATIONS,
    #     OUTPUT_DIR_DATA_STATS,
    #     logger=setup_logger(
    #         OUTPUT_DIR_LOG_CONVERSATIONS,
    #         logger_name="data_stats_extraction.conversation_data_investigation"
    #     )
    # )

## _____________ ##

## CALL OF MAIN FUNCTION ##
if __name__ == "__main__":
    main()