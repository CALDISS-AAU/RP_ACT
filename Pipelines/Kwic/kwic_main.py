"""Main script for the Kwic pipeline.

To run this script, use the following command from the project root:
    uv run python -m Pipelines.Kwic.kwic_main
"""

## IMPORTS ##
from datetime import datetime 
# Internal
from Shared_Functions.logger_functionality import *
from .Functions.generate_kwic_table import generate_kwic_table
from .Functions.summarise_kwic_table import summarise_kwic_table
## _______ ##


## STATIC VARIABLES ##
# Directories - input
INPUT_DIR_CONVERSATION_DATA = "./Data/Clean_Combine/conversation_data_without_advisors.parquet"
INPUT_DIR_MESSAGES_DATA = "./Data/Clean_Combine/message_data_without_advisors.parquet"
INPUT_DIR_ARENAS = "./Data/Derive_Relations_And_Arenas/arenas_comparison.csv"
INPUT_DIR_RELATIONS = "./Data/Derive_Relations_And_Arenas/relations_comparison.csv"

# Directories - internal output
# OUTPUT_DIR_AAA = "Pipelines/Kwic/Data/xxx.zzz"

# Directories - global output
OUTPUT_DIR_KWIC_TABLE_CONVERSATIONS = "Pipelines/Kwic/Data/Full_Sentence/kwic_table_convos.csv"
OUTPUT_DIR_KWIC_SUMMARY_CONVERSATIONS = "Pipelines/Kwic/Data/Full_Sentence/kwic_summary_convos.json"
OUTPUT_DIR_KWIC_TABLE_MESSAGES = "Pipelines/Kwic/Data/Full_Sentence/kwic_table_msgs.csv"
OUTPUT_DIR_KWIC_SUMMARY_MESSAGES = "Pipelines/Kwic/Data/Full_Sentence/kwic_summary_msgs.json"

# Directories - logs
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")
# OUTPUT_DIR_LOG_FULL_PIPELINE = "./Pipelines/Kwic/Logs/full_pipeline.log"
OUTPUT_DIR_LOG_1C = f"./Pipelines/Kwic/Logs/generate_kwic_table_convos_{RUN_TIMESTAMP}.log"
OUTPUT_DIR_LOG_2C = f"./Pipelines/Kwic/Logs/summarise_kwic_table_convos_{RUN_TIMESTAMP}.log"
OUTPUT_DIR_LOG_1M = f"./Pipelines/Kwic/Logs/generate_kwic_table_msgs_{RUN_TIMESTAMP}.log"
OUTPUT_DIR_LOG_2M = f"./Pipelines/Kwic/Logs/summarise_kwic_table_msgs_{RUN_TIMESTAMP}.log"
# OUTPUT_DIR_LOG_2 = "./Pipelines/Kwic/Logs/example_2.log"

# Other
TOP_RELEVANT_ARENAS = 20
TOP_RELEVANT_RELATIONS = 20

TOP_RELEVANT_WORDS = 10

# Relevant-word filtering
RELEVANT_POS = {"NOUN", "ADJ", "VERB"}
EXCLUDE_STOPWORDS = True
ONLY_ALPHABETIC = True
## _______________________ ##


## HELPER FUNCTIONS ##
## ________________ ##


## MAIN FUNCTION ##
def main() -> None:
    """Run the full Kwic pipeline."""
    # logger_1C = setup_logger(
    #     output_dir_log=OUTPUT_DIR_LOG_1C,
    #     logger_name="kwic.step_1C",
    # )

    # generate_kwic_table(
    #     input_dir_data=INPUT_DIR_CONVERSATION_DATA,
    #     input_dir_arenas=INPUT_DIR_ARENAS,
    #     input_dir_relations=INPUT_DIR_RELATIONS,
    #     output_dir=OUTPUT_DIR_KWIC_TABLE_CONVERSATIONS,
    #     top_relevant_arenas=TOP_RELEVANT_ARENAS,
    #     top_relevant_relations=TOP_RELEVANT_RELATIONS,
    #     relevant_pos=RELEVANT_POS,
    #     exclude_stopwords=EXCLUDE_STOPWORDS,
    #     only_alphabetic=ONLY_ALPHABETIC,
    #     logger=logger_1C
    # )

    # logger_2C = setup_logger(
    #     output_dir_log=OUTPUT_DIR_LOG_2C,
    #     logger_name="kwic.step_2C",
    # )

    # kwic_summary = summarise_kwic_table(
    #     input_dir=OUTPUT_DIR_KWIC_TABLE_CONVERSATIONS,
    #     output_dir=OUTPUT_DIR_KWIC_SUMMARY_CONVERSATIONS,
    #     top_m=TOP_RELEVANT_WORDS,
    #     logger=logger_2C,
    # )

    logger_1M = setup_logger(
        output_dir_log=OUTPUT_DIR_LOG_1M,
        logger_name="kwic.step_1M",
    )

    generate_kwic_table(
        input_dir_data=INPUT_DIR_MESSAGES_DATA,
        input_dir_arenas=INPUT_DIR_ARENAS,
        input_dir_relations=INPUT_DIR_RELATIONS,
        output_dir=OUTPUT_DIR_KWIC_TABLE_MESSAGES,
        top_relevant_arenas=TOP_RELEVANT_ARENAS,
        top_relevant_relations=TOP_RELEVANT_RELATIONS,
        relevant_pos=RELEVANT_POS,
        exclude_stopwords=EXCLUDE_STOPWORDS,
        only_alphabetic=ONLY_ALPHABETIC,
        logger=logger_1M
    )

    # logger_2M = setup_logger(
    #     output_dir_log=OUTPUT_DIR_LOG_2M,
    #     logger_name="kwic.step_2M",
    # )

    # kwic_summary = summarise_kwic_table(
    #     input_dir=OUTPUT_DIR_KWIC_TABLE_MESSAGES,
    #     output_dir=OUTPUT_DIR_KWIC_SUMMARY_MESSAGES,
    #     top_m=TOP_RELEVANT_WORDS,
    #     logger=logger_2M,
    # )


## CALL OF MAIN FUNCTION ##
if __name__ == "__main__":
    main()
