""" INTRODUCTION
    Functionality : reads and evaluates .csv datasets for the ACT project
    on the following parameters:
        - Warning when opened?
        - Number of rows and number of conversations
        - Data preview
        - Type determination
        This is all written to a log.
    Application : call the inspect_data function and provide the arguments:
        - input_dir : folder-directory containing the .csv datasets
        - output_dir_log : full path for the log including the name of the log file
"""

## IMPORTS ##
# External
import logging
from pathlib import Path
import pandas as pd
import warnings
## _______ ##

## HELPER FUNCTIONS ##
def _setup_inspection_logger(output_dir_log: str) -> logging.Logger:
    """Set up and return a file logger for data inspection.

    Creates the log directory if it does not exist and configures
    a logger that writes to output_dir_log.

    Args:
        output_dir_log: Full directory where the log file is stored.

    Returns:
        Configured logger instance.
    """
    log_file = Path(output_dir_log)

    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("inspection_of_raw_data")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, mode="w")
        formatter = logging.Formatter("%(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Preview of data
def _data_preview(dataframe: pd.DataFrame, logger: logging.Logger, number: int,):
    """Log example values from each column."""

    logger.info(
        f"Preview of the data ({number} example values per column):"
    )

    for col in dataframe.columns:
        examples = (
            dataframe[col]
            .dropna()
            .unique()
            .tolist()[:number]
        )

        logger.info(f"    {col}: {examples}")

# Data types across columns
def _extract_type_info(dataframe: pd.DataFrame, logger: logging.Logger) -> dict:
    """Extract relevant information from a dataframe 
       and writes it to the log.
    """
    logger.info("The columns within the dataframe and their relevant dtypes are:")

    column_type_info = {}

    for col, dtype in dataframe.dtypes.items():
        logger.info(f"    {col} ({dtype})")

        observed_types = dataframe[col].dropna().map(type).unique()
        observed_type_names = {type_.__name__ for type_ in observed_types}

        column_type_info[col] = {
            "dtype": str(dtype),
            "python_types": observed_type_names,
        }

    logger.info("")

    for col, info in column_type_info.items():
        if len(info["python_types"]) > 1:
            unique_values = dataframe[col].dropna().unique().tolist()

            logger.warning(
                f"Inconsistent types in column '{col}'. "
                f"Observed types: {sorted(info['python_types'])}. "
                f"Unique values: {unique_values}"
            )
    
    return column_type_info

def _summarise_type_info(type_summary: dict, logger: logging.Logger):
    """Summarise observed column types across all inspected datasets."""
    logger.info("TYPE SUMMARY")
    logger.info("Observed dtypes and Python types by column:")

    for col, info in type_summary.items():
        logger.info(f"    {col}:")
        logger.info(f"        Pandas dtypes: {sorted(info['dtypes'])}")
        logger.info(f"        Python types: {sorted(info['python_types'])}")

    logger.info("")

def _update_type_summary(type_summary: dict, column_type_info: dict):
    """Update type summary with type information from one dataframe."""
    for col, info in column_type_info.items():
        if col not in type_summary:
            type_summary[col] = {
                "dtypes": set(),
                "python_types": set(),
            }

        type_summary[col]["dtypes"].add(info["dtype"])
        type_summary[col]["python_types"].update(info["python_types"])

# Summary 
def _summarise_info(summary_info: dict, logger: logging.Logger):
    """Summarise all inspected information across datasets."""
    logger.info("SUMMARY ACROSS ALL DATASETS")
    logger.info("______________________________________________")
    logger.info("")
    
    logger.info(f"There's a total of {summary_info["total_row_count"]} messages across all datasets.")
    logger.info(f"There's a total of {summary_info["total_conversations"]} conversations across all datasets.")

    if "types" in summary_info:
        _summarise_type_info(summary_info["types"], logger)
## ________________ ##

## MAIN FUNCTIONALITY ##
def inspect_data(input_dir: str, output_dir_log: str):
    """Inspect all CSV files in input_dir and write findings to the log."""
    logger = _setup_inspection_logger(output_dir_log)


    summary_info = {
        "total_row_count": 0,
        "total_conversations": 0,
        "types": {},
    }

    for file in Path(input_dir).rglob("*.csv"):
        logger.info(f"{file.name} has been found.")

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            dataframe = pd.read_csv(file, sep=";")

        for warning in caught_warnings:
            logger.warning(f"WARNING: {warning.message}")

        logger.info("The dataset was successfully opened.")

        # Number of rows / messages
        row_count = dataframe.shape[0]
        summary_info["total_row_count"] = summary_info["total_row_count"] + row_count
        logger.info(f"This dataset contain {row_count} rows.")

        # Number of conversations
        conversation_count = len(dataframe["conversation_code"].unique())
        summary_info["total_conversations"] = summary_info["total_conversations"] + conversation_count
        logger.info(f"This dataset contain {conversation_count} conversations.")

        # Preview
        _data_preview(dataframe, logger, 3)

        # Types
        column_type_info = _extract_type_info(dataframe, logger)
        _update_type_summary(summary_info["types"], column_type_info)

        logger.info("The evaluation of this dataset has been finished!")
        logger.info("______________________________________________")
        logger.info("")

    # Summary
    _summarise_info(summary_info, logger)
## __________________ ##
