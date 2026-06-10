## IMPORTS ##
# External
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow
# Internal
from .data_inspection import inspect_cleaned_data
## _______ ##

## HELPER FUNCTIONS ##
def _setup_cleaning_logger(output_dir_log: str) -> logging.Logger:
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

def _combine_data(input_dir: str, logger):
    dfs = []
    for file in Path(input_dir).rglob("*.csv"):
        dfs.append(pd.read_csv(file, sep=";"))

    logger.info(f"There's a total of {len(dfs)} files in total.")

    df_combined = pd.concat(dfs, ignore_index=True)
    return df_combined

def _remove_empty_texts(dataframe: pd.DataFrame, logger):
    dataframe_non_empty_text = dataframe[
        dataframe["msg_tekst"].replace(r"^\s*$", np.nan, regex=True)
        .notna()
    ]
    logger.info("All empty messages has been removed.")

    return dataframe_non_empty_text

def _remove_regular_users(dataframe: pd.DataFrame, logger):
    dataframe_non_regulars = dataframe[
        dataframe["conversation_type"] == "Rådgivningssamtale"
    ]
    logger.info("All regular users has been removed")

    return dataframe_non_regulars

def _remove_ages_out_of_range(dataframe: pd.DataFrame, logger):
    dataframe_in_range_ages = dataframe[
        ~dataframe["requester_age_name"].astype(str).str.contains(r"\+", na=False)
    ]

    dataframe_in_range_ages["requester_age_name"] = pd.to_numeric(
        dataframe_in_range_ages["requester_age_name"], errors="coerce"
    )

    dataframe_in_range_ages = dataframe_in_range_ages[
        dataframe_in_range_ages["requester_age_name"].between(10, 17)
    ]

    dataframe_in_range_ages["requester_age_name"] = (
        dataframe_in_range_ages["requester_age_name"].astype(int)
    )
    logger.info("All ages out of range has been removed")

    return dataframe_in_range_ages

def _clean_data(dataframe: pd.DataFrame, logger):
    dataframe_step1 = _remove_empty_texts(dataframe, logger)
    dataframe_step2 = _remove_regular_users(dataframe_step1, logger)
    dataframe_step3 = _remove_ages_out_of_range(dataframe_step2, logger)
    dataframe_step3["is_incoming"] = dataframe_step3["is_incoming"].astype(bool)
    logger.info("'is_incoming' is now a bool")
    dataframe_step3["last_contact"] = pd.to_datetime(dataframe_step3["last_contact"], utc=True)
    dataframe_step3["msg_created_on"] = pd.to_datetime(dataframe_step3["msg_created_on"], utc=True)
    logger.info("'last_contact' and 'msg_created_on' are now datetimes")

    clean_dataframe = dataframe_step3
    return clean_dataframe
## ________________ ##

## MAIN FUNCTIONALITY ##
def clean_and_combine_data(input_dir: str, output_dir_combined_data: str, output_dir_log: str):
    """Clean raw data to make it polars compatible."""
    logger = _setup_cleaning_logger(output_dir_log)

    df_combined = _combine_data(input_dir, logger)

    # Pre clean logging
    logger.info("All datasets have been combined!")
    logger.info("_____________________________________________________")
    logger.info("")
    logger.info("INFO: Pre clean logging:")
    logger.info(f"All unique is_incoming: {df_combined["is_incoming"].unique()}")
    logger.info(f"All unique ages: {df_combined["requester_age_name"].unique()}")
    logger.info(f"All unique conversation_type: {df_combined["conversation_type"].unique()}")
    logger.info(f"Rows where 'is_incoming'=nan: {df_combined[~df_combined["is_incoming"].isin([True, False])]}")
    logger.info("_____________________________________________________")
    logger.info("")

    df_clean = _clean_data(df_combined, logger)

    # Post clean logging
    logger.info("All datasets have been cleaned!")
    logger.info("_____________________________________________________")
    logger.info("")
    logger.info("INFO: Post clean logging:")
    logger.info(f"All unique is_incoming: {df_clean["is_incoming"].unique()}")
    logger.info(f"All unique ages: {df_clean["requester_age_name"].unique()}")
    logger.info(f"All unique conversation_type: {df_clean["conversation_type"].unique()}")
    logger.info(f"Rows where 'is_incoming'=nan: {df_clean[~df_clean["is_incoming"].isin([True, False])]}")
    logger.info("_____________________________________________________")
    logger.info("")

    inspect_cleaned_data(df_clean, logger)

    df_clean.to_parquet(output_dir_combined_data)
    # df_clean.to_json(output_dir_combined_data, orient="records", lines=True)

    
## __________________ ##
