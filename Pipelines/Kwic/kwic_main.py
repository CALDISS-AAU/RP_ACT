"""Main script for the Kwic pipeline.

To run this script, use the following command from the project root:
    uv run python -m Pipelines.Kwic.kwic_main
"""

## IMPORTS ##
# Internal
from Shared_Functions.logger_functionality import *
from .Functions.generate_kwic_table import generate_kwic_table
## _______ ##


## STATIC VARIABLES ##
# Directories - input
INPUT_DIR_CONVERSATION_DATA = "./Data/Clean_Combine/collapsed_messages_data_without_advisors.parquet"
INPUT_DIR_ARENAS = "./Data/Derive_Relations_And_Arenas/arenas_comparison.csv"
INPUT_DIR_RELATIONS = "./Data/Derive_Relations_And_Arenas/relations_comparison.csv"

# Directories - internal output
# OUTPUT_DIR_AAA = "Pipelines/Kwic/Data/xxx.zzz"

# Directories - global output
OUTPUT_DIR_KWIC_TABLE = "./Data/Kwic/kwic_table.csv"

# Directories - logs
# OUTPUT_DIR_LOG_FULL_PIPELINE = "./Pipelines/Kwic/Logs/full_pipeline.log"
OUTPUT_DIR_LOG_1 = "./Pipelines/Kwic/Logs/generate_kwic_table.log"
# OUTPUT_DIR_LOG_2 = "./Pipelines/Kwic/Logs/example_2.log"

# Other
words_before_and_after_keyword = 4
top_relevant_arenas = 20
top_relevant_relations = 20
## _______________________ ##


## HELPER FUNCTIONS ##
## ________________ ##


## MAIN FUNCTION ##
def main() -> None:
    """Run the full Kwic pipeline."""
    logger_1 = setup_logger(
        output_dir_log=OUTPUT_DIR_LOG_1,
        logger_name="kwic.step_1",
    )

    logger_1.info("In the beginning there was darkness.........")
    print("I have printed stuff!! Yay me!")

    generate_kwic_table(
        input_dir_data=INPUT_DIR_CONVERSATION_DATA,
        input_dir_arenas=INPUT_DIR_ARENAS,
        input_dir_relations=INPUT_DIR_RELATIONS,
        output_dir=OUTPUT_DIR_KWIC_TABLE,
        words_before_and_after_keyword=words_before_and_after_keyword,
        top_relevant_arenas=top_relevant_arenas,
        top_relevant_relations=top_relevant_relations,
        logger=logger_1
    )

    # example_function(
    #     input_str="Hello",
    #     logger=setup_logger(
    #         output_dir_log=OUTPUT_DIR_LOG_1,
    #         logger_name="kwic.step_1",
    #     ),
    # )

    # example_function(
    #     input_str="World!",
    #     logger=setup_logger(
    #         output_dir_log=OUTPUT_DIR_LOG_2,
    #         logger_name="kwic.step_2",
    #     ),
    # )

    # rebuild_pipeline_log(
    #     step_log_paths=[
    #         OUTPUT_DIR_LOG_1,
    #         OUTPUT_DIR_LOG_2,
    #     ],
    #     output_dir_log=OUTPUT_DIR_LOG_FULL_PIPELINE,
    # )


## CALL OF MAIN FUNCTION ##
if __name__ == "__main__":
    main()
