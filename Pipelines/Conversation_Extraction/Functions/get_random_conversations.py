"""Export random full conversation examples from conversation data."""

## IMPORTS ##
# Standard
import logging
from pathlib import Path

# External
import polars as pl
## _______ ##


## HELPER FUNCTIONS ##
def _get_random_conversations(
    df: pl.DataFrame,
    n_conversations: int,
    logger: logging.Logger,
) -> pl.DataFrame:
    """Sample random conversations with non-empty message text."""
    logger.info(f"Sampling {n_conversations} random conversations.")

    valid_df = df.filter(
        pl.col("msg_tekst").is_not_null()
        & (pl.col("msg_tekst").str.strip_chars() != "")
    )

    if valid_df.height < n_conversations:
        logger.warning(
            f"Requested {n_conversations} conversations, "
            f"but only {valid_df.height} valid conversations are available."
        )

        n_conversations = valid_df.height

    return (
        valid_df
        .sample(n=n_conversations, shuffle=True)
        .select(
            "conversation_code",
            "gender",
            "age_group",
            "msg_tekst",
        )
    )


def _save_conversations_to_txt(
    df: pl.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    """Save each sampled conversation to a separate text file."""
    logger.info(
        f"Saving {df.height} conversation examples to {output_dir}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(df.iter_rows(named=True), start=1):
        output_path = output_dir / f"conversation_{i}.txt"

        with output_path.open("w", encoding="utf-8") as file:
            file.write(f"conversation_code: {row['conversation_code']}\n")
            file.write(f"gender: {row['gender']}\n")
            file.write(f"age_group: {row['age_group']}\n")
            file.write("\n")
            file.write(str(row["msg_tekst"]))

        logger.info(f"Saved {output_path}")
## ________________ ##


## MAIN FUNCTIONALITY ##
def export_random_conversation_examples(
    input_dir_data: str,
    output_dir_folder: str,
    n_conversations: int,
    logger: logging.Logger,
) -> None:
    """Export random full conversation examples as text files.
       Reads the conversation-level dataset, samples conversations with
           non-empty message text, and saves each selected conversation as
           a separate text file with basic metadata.

       Args:
           input_dir_data: Path to the conversation-level parquet dataset.
           output_dir_folder: Directory where conversation text files are saved.
           n_conversations: Number of random conversations to export.
           logger: Logger used to write processing information.
    """
    logger.info(f"Reading conversation data from {input_dir_data}")

    df = pl.read_parquet(input_dir_data)

    logger.info(f"Loaded dataset with {df.height} rows and {df.width} columns.")

    random_conversations = _get_random_conversations(
        df=df,
        n_conversations=n_conversations,
        logger=logger,
    )

    _save_conversations_to_txt(
        df=random_conversations,
        output_dir=Path(output_dir_folder),
        logger=logger,
    )

    logger.info("Finished exporting random conversation examples.")
## __________________ ##