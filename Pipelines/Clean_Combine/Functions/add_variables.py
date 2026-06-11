## IMPORTS ##
# External
import logging
from pathlib import Path
import polars as pl
import pyarrow
import json
# Internal
## _______ ##

## HELPER FUNCTIONS ##
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

def _open_and_combine_contact_reasons(input_dir: str, logger):
    dfs = []
    for file in Path(input_dir).rglob("*.csv"):
        dfs.append(pl.read_csv(file, separator=";"))
    
    df_combined = pl.concat(dfs)
    return df_combined

def _merge_topic_row(row: dict) -> list[str]:
    topics = []

    for cell in row.values():
        if cell is None:
            continue

        cell_topics = json.loads(cell)

        for topic in cell_topics:
            topics.append(topic)

    return topics

def _merge_topics(dataframe: pl.DataFrame, logger: logging.Logger):
    """Merges all topics into a single list"""
    topic_cols = []
    for col in dataframe.columns:
        if "topic_" in col:
            topic_cols.append(col)

    dataframe = dataframe.with_columns(
        pl.struct(topic_cols)
        .map_elements(
            _merge_topic_row,
            return_dtype=pl.List(pl.String),
        )
        .alias("contact_reasons")
    )
    
    logger.info(f"All topics from the following columns have been merged:")
    logger.info(f"{topic_cols}")

    return dataframe

def _add_contact_reason(dataframe_data: pl.DataFrame, dataframe_contact_reason: pl.DataFrame, logger: logging.Logger):
    """Adds column with contact reason from extern dataset"""
    dataframe_contact_reason = _merge_topics(dataframe_contact_reason, logger)

    dataframe_data = dataframe_data.join(
        dataframe_contact_reason.select([
            "conversation_code", 
            "contact_reasons"
        ]),
        on="conversation_code",
        how="left"
    )

    return dataframe_data
    

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
    logger: logging.Logger,
    age_range_dict: dict,
    time_of_day_dict: dict,
    season_dict: dict,
) -> None:
    """Adds relevant variables to the dataset."""

    df = pl.read_parquet(input_dir_data)
    df_contact_reason = _open_and_combine_contact_reasons(input_dir_contact_reason, logger)

    logger.info("A summary of the initial data")
    _give_summary(df, logger)
    logger.info("_____________________________________________________")
    logger.info("")

    logger.info("A summary of the contact_reason data")
    _give_summary(df_contact_reason, logger)
    logger.info("_____________________________________________________")
    logger.info("")

    df = _add_age_ranges(df, "requester_age_name", age_range_dict, logger)
    df = _add_time_of_day(df, "msg_created_on", time_of_day_dict, logger)
    df = _add_season(df, "msg_created_on", season_dict, logger)
    df = _add_text_features(df, "msg_tekst", logger)
    logger.info("_____________________________________________________")
    logger.info("")

    df = _add_contact_reason(df, df_contact_reason, logger)
    logger.info("_____________________________________________________")
    logger.info("")

    logger.info("A summary of the final data")
    _give_summary(df, logger)
    logger.info("_____________________________________________________")
    logger.info("")

    df.write_parquet(output_dir_data)
## __________________ ##
