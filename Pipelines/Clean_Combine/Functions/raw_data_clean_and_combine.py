"""Combine and clean ACT conversation datasets for downstream analysis.
   The module reads raw conversation CSV files, combines them into a
       single dataframe, applies project-specific cleaning rules,
       inspects the cleaned result, and exports it as parquet.
"""

## IMPORTS ##
# Standard
import logging
from pathlib import Path

# External
import numpy as np
import pandas as pd
import pyarrow

# Internal
from .data_inspection import inspect_cleaned_data
## _______ ##

## HELPER FUNCTIONS ##
def _combine_data(input_dir: str, logger: logging.Logger) -> pd.DataFrame: 
    """Read and combine all CSV files from a directory."""

    dfs = []
    for file in Path(input_dir).rglob("*.csv"):
        df = pd.read_csv(file, sep=";")
        logger.info(f'This dataset ({file}) contains the following columns: {df.columns}')
        if 'Kolonne1' in df.columns:
            logger.info(f'All unique Kolonne1 values: {df['Kolonne1'].unique()}')
            df = df.drop(columns=['Kolonne1'])
            logger.info(f'All columns of {file} after dropping Kolonne1: {df.columns}')
        dfs.append(df)

    logger.info(f"There's a total of {len(dfs)} files in total.")

    df_combined = pd.concat(dfs, ignore_index=True)
    return df_combined

def _remove_empty_texts(dataframe: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Remove rows with empty or whitespace-only message text."""

    rows_before = len(dataframe)

    dataframe_non_empty_text = dataframe[
        dataframe["msg_tekst"].replace(r"^\s*$", np.nan, regex=True)
        .notna()
    ]

    rows_removed = rows_before - len(dataframe_non_empty_text)

    logger.info("All empty messages have been removed.")
    logger.info(f"{rows_removed} messages were removed in this step.")

    return dataframe_non_empty_text

def _remove_regular_users(dataframe: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Remove messages from/to regular users."""

    rows_before = len(dataframe)

    dataframe_non_regulars = dataframe[
        dataframe["conversation_type"] == "Rådgivningssamtale"
    ]

    rows_removed = rows_before - len(dataframe_non_regulars)

    logger.info("All regular users have been removed")
    logger.info(f"{rows_removed} messages were removed in this step.")

    return dataframe_non_regulars

def _remove_ages_out_of_range(dataframe: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Keep only requester ages from 10 to 17."""

    rows_before = len(dataframe)

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

    rows_removed = rows_before - len(dataframe_in_range_ages)
    
    logger.info("All ages out of range have been removed")
    logger.info(f"{rows_removed} messages were removed in this step.")

    return dataframe_in_range_ages

def _clean_data(dataframe: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Apply filtering and type conversions to the combined dataframe."""

    map = {
        "SAND": True,
        "FALSK": False
    }
    dataframe["is_incoming"] = dataframe["is_incoming"].map(map)
    dataframe_step1 = _remove_empty_texts(dataframe, logger)
    dataframe_step2 = _remove_regular_users(dataframe_step1, logger)
    dataframe_step3 = _remove_ages_out_of_range(dataframe_step2, logger)
    dataframe_step3["is_incoming"] = dataframe_step3["is_incoming"].astype(bool)
    logger.info("'is_incoming' is now a bool")
    dataframe_step3["last_contact"] = pd.to_datetime(dataframe_step3["last_contact"], format="%d-%m-%Y %H:%M", utc=True)
    dataframe_step3["msg_created_on"] = pd.to_datetime(dataframe_step3["msg_created_on"], format="%d-%m-%Y %H:%M", utc=True)
    logger.info("'last_contact' and 'msg_created_on' are now datetimes")

    clean_dataframe = dataframe_step3
    return clean_dataframe
## ________________ ##

## MAIN FUNCTIONALITY ##
def clean_and_combine_data(
    input_dir: str,
    output_dir_combined_data: str,
    cleaning_logger: logging.Logger,
    evaluation_logger: logging.Logger
) -> None:
    """Combine, clean, inspect, and export raw ACT CSV datasets. 
       Reads all CSV files in ``input_dir``, combines them into one dataframe, 
           and applies project-specific cleaning rules. 
       The function logs values before and after cleaning, 
           inspects the cleaned dataframe, and exports the final result 
           as a parquet file. 
    
       Args: 
           input_dir: Directory containing raw CSV files. 
           output_dir_combined_data: Output path for the cleaned parquet file. 
           cleaning_logger: Logger used for cleaning and transformation messages. 
           evaluation_logger: Logger used for cleaned-data inspection results. 
    """

    df_combined = _combine_data(input_dir, cleaning_logger)

    # Pre clean logging
    cleaning_logger.info("All datasets have been combined!")
    cleaning_logger.info("_____________________________________________________")
    cleaning_logger.info("")
    cleaning_logger.info("INFO: Pre clean logging:")
    cleaning_logger.info(f"All unique is_incoming: {df_combined['is_incoming'].unique()}")
    cleaning_logger.info(f"All unique ages: {df_combined['requester_age_name'].unique()}")
    cleaning_logger.info(f"All unique conversation_type: {df_combined['conversation_type'].unique()}")
    cleaning_logger.info(f"Rows where 'is_incoming'=nan: {df_combined[~df_combined['is_incoming'].isin([True, False])]}")
    cleaning_logger.info("_____________________________________________________")
    cleaning_logger.info("")

    df_clean = _clean_data(df_combined, cleaning_logger)

    # Post clean logging
    cleaning_logger.info("All datasets have been cleaned!")
    cleaning_logger.info("_____________________________________________________")
    cleaning_logger.info("")
    cleaning_logger.info("INFO: Post clean logging:")
    cleaning_logger.info(f"All unique is_incoming: {df_clean['is_incoming'].unique()}")
    cleaning_logger.info(f"All unique ages: {df_clean['requester_age_name'].unique()}")
    cleaning_logger.info(f"All unique conversation_type: {df_clean['conversation_type'].unique()}")
    cleaning_logger.info(f"Rows where 'is_incoming'=nan: {df_clean[~df_clean['is_incoming'].isin([True, False])]}")
    cleaning_logger.info("_____________________________________________________")
    cleaning_logger.info("")

    inspect_cleaned_data(df_clean, evaluation_logger)

    df_clean.to_parquet(output_dir_combined_data)
    # df_clean.to_json(output_dir_combined_data, orient="records", lines=True)

    
## __________________ ##
