"""Generate additional working datasets from cleaned ACT message data.
   The module creates alternative representations of the cleaned message
       dataset for downstream analyses. It generates a dataset with
       consecutive messages collapsed by sender and a conversation-level
       dataset where all messages are combined into a single row.
"""

## IMPORTS ##
# Standard
import logging

# External
import polars as pl
import pyarrow
## _______ ##

## HELPER FUNCTIONS ##
def _find_random_conversation_code(
    dataframe: pl.DataFrame,
    logger: logging.Logger,
    minimum_messages: int,
) -> str:
    """Find a random conversation with at least a given number of messages."""

    conversation_code = (
        dataframe
        .filter(pl.col("msg_nummer") >= minimum_messages)
        .sample(n=1)
        .select("conversation_code")
        .item()
    )

    logger.info(f"For validation the conversation {conversation_code} has been randomly chosen.")
    return conversation_code

def _data_preview(
    dataframe: pl.DataFrame,
    logger: logging.Logger,
    conversation_code: str,
) -> None:
    """Log all values for a single conversation."""

    dataframe = (
        dataframe
        .filter(pl.col("conversation_code") == conversation_code)
        .sort("msg_nummer")
    )

    logger.info(f"Preview of conversation: {conversation_code}")
    logger.info(f"Rows: {dataframe.height}")

    for col in dataframe.columns:
        values = dataframe[col].to_list()

        logger.info(f"{col}: {values}")

    logger.info("")
    
def _collapse_messages(
    dataframe: pl.DataFrame,
    logger: logging.Logger,
    separator: str,
) -> pl.DataFrame:
    """Combine consecutive messages from the same sender into a single message."""

    dataframe = dataframe.sort(["conversation_code", "msg_nummer"])

    collapsed_rows = []

    current_row = None
    current_conversation = None
    current_is_incoming = None
    text_block = []

    for row in dataframe.iter_rows(named=True):
        new_block = (
            current_row is None
            or row["conversation_code"] != current_conversation
            or row["is_incoming"] != current_is_incoming
        )

        if new_block:
            if current_row is not None:
                current_row["msg_tekst"] = separator.join(text_block)
                collapsed_rows.append(current_row)

            current_row = row
            current_conversation = row["conversation_code"]
            current_is_incoming = row["is_incoming"]
            text_block = [row["msg_tekst"]]
        else:
            text_block.append(row["msg_tekst"])

    if current_row is not None:
        current_row["msg_tekst"] = separator.join(text_block)
        collapsed_rows.append(current_row)

    dataframe = pl.DataFrame(collapsed_rows)

    logger.info("Consecutive messages have been collapsed.")

    return dataframe

def _combine_conversations(
    dataframe: pl.DataFrame,
    logger: logging.Logger,
    separator: str,
) -> pl.DataFrame:
    """Combine all messages in each conversation into one row."""

    dataframe = dataframe.sort(["conversation_code", "msg_nummer"])

    aggregations = []

    for col in dataframe.columns:
        if col == "conversation_code":
            continue

        if col == "msg_tekst":
            aggregations.append(
                pl.col(col).str.concat(separator).alias(col)
            )
        elif col == "is_incoming":
            aggregations.append(
                pl.lit(None).alias(col)
            )
        else:
            aggregations.append(
                pl.col(col).first().alias(col)
            )

    dataframe = (
        dataframe
        .group_by("conversation_code", maintain_order=True)
        .agg(aggregations)
    )

    logger.info("Messages have been combined into conversation-level rows.")

    return dataframe

## ________________ ##

## MAIN FUNCTIONALITY ##
def generate_working_datasets(
    input_dir_data: str,
    output_dir_collapsed_messages_data: str,
    output_dir_conversation_data: str,
    logger: logging.Logger,
    separator: str,
) -> None:
    """Generate additional working datasets from cleaned message data.
       Reads the cleaned message-level dataset and creates two additional
           datasets for downstream analyses. One dataset collapses
           consecutive messages from the same sender, while the other
           combines all messages within each conversation into a single row.
           A randomly selected conversation is logged before and after each
           transformation for validation, and the generated datasets are
           exported as parquet files.

       Args:
           input_dir_data: Path to the cleaned message-level parquet dataset.
           output_dir_collapsed_messages_data: Output path for the dataset
               with consecutive messages collapsed.
           output_dir_conversation_data: Output path for the conversation-level
               dataset.
           logger: Logger used to write processing information.
           separator: String used when combining message texts.
    """
    df = pl.read_parquet(input_dir_data)
    random_conversation = _find_random_conversation_code(df, logger, 20)

    _data_preview(df, logger, random_conversation)

    df_collapsed_messages = _collapse_messages(df, logger, separator)
    _data_preview(df_collapsed_messages, logger, random_conversation)

    df_conversations = _combine_conversations(df, logger, separator)
    _data_preview(df_conversations, logger, random_conversation)

    df_collapsed_messages.write_parquet(output_dir_collapsed_messages_data)
    df_conversations.write_parquet(output_dir_conversation_data)

    
## __________________ ##
