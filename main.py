""" This is the main script of the ACT-project.
    This script is reserved for calling and combining pipelines from the
        Pipelines/ folders.
    For a description of the overall functionality of the individual
        pipelines, please consult their respective Pipelines/README.md.
"""

""" Run using
        uv run python -m main
"""
## IMPORTS ##
# Pipeline mains
from Pipelines.Clean_Combine.clean_combine_main import main as clean_combine_main
from Pipelines.Conversation_Extraction.conversation_extraction_main import main as conversation_extraction_main
from Pipelines.Data_Stats_Extraction.data_stats_extraction_main import main as data_stats_extraction_main
## _______ ##

## MAIN FUNCTION ##
def main() -> None:
    clean_combine_main()
    conversation_extraction_main()
    data_stats_extraction_main()
## _____________ ##

## CALL OF MAIN FUNCTION ##
if __name__ == "__main__":
    main()