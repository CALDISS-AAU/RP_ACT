"""Main script for the Data_Stats_Extraction pipeline.

Run the ACT data statistics extraction pipeline.
The script generates descriptive statistics for the cleaned ACT datasets,
including the original messages, collapsed messages, and conversations.
The resulting statistics are exported to a shared JSON file, and a
pipeline log is rebuilt from the logs of the individual processing steps.

Run from the ACT folder with:
    uv run python -m Pipelines.Data_Stats_Extraction.data_stats_extraction_main
"""

## IMPORTS ##
# Internal
from Shared_Functions.logger_functionality import *
from .Functions.data_investigation import *

## _______ ##

## STATIC VARIABLES ##
# Directories - input
INPUT_DIR_BASE = "./Data/Clean_Combine"
INPUT_DIR_MESSAGES = f"{INPUT_DIR_BASE}/message_data.parquet" 
INPUT_DIR_COLLAPSED_MESSAGES = f"{INPUT_DIR_BASE}/collapsed_messages_data.parquet" 
INPUT_DIR_CONVERSATIONS = f"{INPUT_DIR_BASE}/conversation_data.parquet" 

# Directories - global output
OUTPUT_DIR_DATA_STATS = "./Data/Data_Stats_Extraction/data_stats.json" 

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

    data_investigation_collapsed_messages(
        INPUT_DIR_COLLAPSED_MESSAGES,
        OUTPUT_DIR_DATA_STATS,
        logger=setup_logger(
            OUTPUT_DIR_LOG_COLLAPSED_MESSAGES,
            logger_name="data_stats_extraction.collapsed_message_data_investigation"
        )
    )

    data_investigation_conversations(
        INPUT_DIR_CONVERSATIONS,
        OUTPUT_DIR_DATA_STATS,
        logger=setup_logger(
            OUTPUT_DIR_LOG_CONVERSATIONS,
            logger_name="data_stats_extraction.conversation_data_investigation"
        )
    )

    rebuild_pipeline_log(
        step_log_paths=[
            OUTPUT_DIR_LOG_MESSAGES,
            OUTPUT_DIR_LOG_COLLAPSED_MESSAGES,
            OUTPUT_DIR_LOG_CONVERSATIONS,
        ],
        output_dir_log=OUTPUT_DIR_LOG_FULL_PIPELINE,
    )

## _____________ ##

## CALL OF MAIN FUNCTION ##
if __name__ == "__main__":
    main()