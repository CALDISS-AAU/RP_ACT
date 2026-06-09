## IMPORTS ##
# External
import logging
from pathlib import Path
import polars as pl
import pyarrow
# Internal
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

def _dict_to_mapping(dict):
    mapping = {
        key: values
        for values, keys in dict.items()
        for key in keys
    }

    return mapping

def _add_age_ranges(
    dataframe: pl.DataFrame,
    col: str,
    age_ranges: dict,
    logger: logging.Logger,
) -> pl.DataFrame:
    """Adds column with age group indicator.
       Takes a dict of age ranges as input.
    """
    age_mapping = _dict_to_mapping(age_ranges)

    dataframe = dataframe.with_columns(
        pl.col(col)
        .replace_strict(age_mapping, return_dtype=pl.String)
        .alias("age_group")
    )

    logger.info("Age group has been added")
    return dataframe

def _add_text_features(
    dataframe: pl.DataFrame,
    col: str,
    logger: logging.Logger,
) -> pl.DataFrame:
    """Adds text length features."""

    dataframe = dataframe.with_columns(
        pl.col(col)
        .str.len_chars()
        .alias("text_characters"),

        pl.col(col)
        .str.extract_all(r"\S+")
        .list.len()
        .alias("text_words"),
    )

    dataframe = dataframe.with_columns(
        pl.when(pl.col("text_words") > 0)
        .then(
            (pl.col("text_characters") / pl.col("text_words"))
            .round(2)
        )
        .otherwise(None)
        .alias("avg_word_length")
    )

    logger.info(f"Text features have been added based on column: {col}")
    return dataframe

def _add_time_of_day(
    dataframe: pl.DataFrame,
    col: str,
    time_ranges: dict,
    logger: logging.Logger,
) -> pl.DataFrame:
    """Adds column with time of day indicator
       Takes a dict of time ranges as input
    """
    time_mapping = _dict_to_mapping(time_ranges)

    dataframe = dataframe.with_columns(
        pl.col(col).dt.hour()
        .replace_strict(time_mapping, return_dtype=pl.String)
        .alias("time_of_day")
    )

    logger.info("Time of day has been added")
    return dataframe

def _add_season(
    dataframe: pl.DataFrame,
    col: str,
    season_ranges: dict,
    logger: logging.Logger,
) -> pl.DataFrame:
    """Adds column with season indicator
       Takes a dict of year ranges as input
    """
    season_mapping = _dict_to_mapping(season_ranges)

    dataframe = dataframe.with_columns(
        pl.col(col).dt.month()
        .replace_strict(season_mapping, return_dtype=pl.String)
        .alias("season")
    )

    logger.info("Season has been added")
    return dataframe

def _add_contact_reason(dataframe: pl.DataFrame, logger: logging.Logger):
    """Adds column with contact reason from extern dataset"""

def _give_summary(dataframe: pl.DataFrame, logger: logging.Logger):
    logger.info(f"Shape: {dataframe.shape}")

    for col in dataframe.columns:
        logger.info(
            f"{col} | {dataframe[col].dtype} | "
            f"{dataframe[col].unique().head(5).to_list()}"
        )
## ________________ ##

## MAIN FUNCTIONALITY ##
def add_variables(
    input_dir_data: str, 
    input_dir_contact_reason: str,
    output_dir_data: str, 
    output_dir_log: str,
    age_range_dict: dict,
    time_of_day_dict: dict,
    season_dict: dict,
):
    """Adds relevant variables to the dataset."""
    logger = _setup_cleaning_logger(output_dir_log)

    df = pl.read_parquet(input_dir_data)
    logger.info("A summary of the initial data")
    _give_summary(df, logger)
    logger.info("_____________________________________________________")
    logger.info("")

    df = _add_age_ranges(df, "requester_age_name", age_range_dict, logger)
    df = _add_time_of_day(df, "msg_created_on", time_of_day_dict, logger)
    df = _add_season(df, "msg_created_on", season_dict, logger)
    df = _add_text_features(df, "msg_tekst", logger)
    logger.info("_____________________________________________________")
    logger.info("")

    logger.info("A summary of the final data")
    _give_summary(df, logger)
    logger.info("_____________________________________________________")
    logger.info("")
## __________________ ##
