"""Inspect ACT conversation datasets and log basic quality information.
   The module supports inspection of raw CSV files and cleaned dataframes.
   It logs row counts, conversation counts, column previews, pandas dtypes,
       observed Python types, and summary information across datasets.
"""

## IMPORTS ##
# Standard
import logging
from pathlib import Path
import warnings

# External
import pandas as pd
## _______ ##

## HELPER FUNCTIONS ##
# Preview of data
def _data_preview(dataframe: pd.DataFrame, logger: logging.Logger, number: int,) -> None:
    """Log example values from each dataframe column."""

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

def _extract_unique_values(
    dataframe: pd.DataFrame,
    logger: logging.Logger,
    column_name: str,
) -> set:
    """Log and return unique non-null values from one column."""

    if column_name not in dataframe.columns:
        logger.warning(f"Column '{column_name}' was not found in this dataset.")
        return set()

    unique_values = set(dataframe[column_name].dropna().unique().tolist())

    logger.info(f"Unique values in '{column_name}': {sorted(unique_values)}")

    return unique_values

# Data types across columns
def _extract_type_info(dataframe: pd.DataFrame, logger: logging.Logger) -> dict:
    """Log dtype information and return observed column type metadata."""

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

def _summarise_type_info(type_summary: dict, logger: logging.Logger) -> None:
    """Log observed dtypes and Python types across inspected datasets."""

    logger.info("TYPE SUMMARY")
    logger.info("Observed dtypes and Python types by column:")

    for col, info in type_summary.items():
        logger.info(f"    {col}:")
        logger.info(f"        Pandas dtypes: {sorted(info['dtypes'])}")
        logger.info(f"        Python types: {sorted(info['python_types'])}")

    logger.info("")

def _update_type_summary(type_summary: dict, column_type_info: dict) -> None: 
    """Add one dataframe's column type metadata to the overall summary."""

    for col, info in column_type_info.items():
        if col not in type_summary:
            type_summary[col] = {
                "dtypes": set(),
                "python_types": set(),
            }

        type_summary[col]["dtypes"].add(info["dtype"])
        type_summary[col]["python_types"].update(info["python_types"])

# Summary 
def _summarise_info(summary_info: dict, logger: logging.Logger) -> None: 
    """Log summary statistics across all inspected datasets."""

    logger.info("SUMMARY ACROSS ALL DATASETS")
    logger.info("______________________________________________")
    logger.info("")
    
    logger.info(f"There's a total of {summary_info['total_row_count']} messages across all datasets.")
    logger.info(f"There's a total of {summary_info['total_conversations']} conversations across all datasets.")

    if "types" in summary_info:
        _summarise_type_info(summary_info["types"], logger)

    if "requester_gender_name_values" in summary_info:
        logger.info(
            "Unique requester_gender_name values across all datasets: "
            f"{sorted(summary_info['requester_gender_name_values'])}"
        )
## ________________ ##

## MAIN FUNCTIONALITY ##
def inspect_raw_data(input_dir: str, logger: logging.Logger) -> None:
    """Inspect all CSV files in a directory and log dataset-level findings. 
       Recursively searches ``input_dir`` for CSV files, opens each file 
           using a semicolon separator, and logs warnings raised during import. 
       For each dataset, the function logs row counts, conversation counts, 
           example values, column dtypes, and observed Python types. 
       It finishes by logging a combined summary across all inspected datasets. 
       
       Args: 
           input_dir: Directory containing raw CSV files. 
           logger: Logger used to write inspection results. 
    """

    summary_info = {
        "total_row_count": 0,
        "total_conversations": 0,
        "types": {},
        "requester_gender_name_values": set(),
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

        # Gender
        gender_values = _extract_unique_values(
            dataframe,
            logger,
            "requester_gender_name",
        )

        summary_info["requester_gender_name_values"].update(gender_values)

    # Summary
    _summarise_info(summary_info, logger)

def inspect_cleaned_data(dataframe: pd.DataFrame, logger: logging.Logger)-> None: 
    """Inspect a cleaned dataframe and log basic dataset information.
       Logs row counts, conversation counts, example values, column dtypes,
           observed Python types, and selected unique values from the cleaned
           dataframe.

       Args:
           dataframe: Cleaned dataframe to inspect.
           logger: Logger used to write inspection results.
    """

    logger.info("INSPECTION OF CLEANED DATA")
    logger.info("______________________________________________")

    row_count = dataframe.shape[0]
    logger.info(f"This dataset contains {row_count} rows.")

    conversation_count = dataframe["conversation_code"].nunique()
    logger.info(f"This dataset contains {conversation_count} conversations.")

    _data_preview(dataframe, logger, 3)

    _extract_type_info(dataframe, logger)

    _extract_unique_values(dataframe, logger, "requester_gender_name")

    logger.info("The evaluation of the cleaned dataset has been finished!")
    logger.info("______________________________________________")
    logger.info("")
## __________________ ##
