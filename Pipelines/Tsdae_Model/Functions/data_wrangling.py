# External
from pathlib import Path
import polars as pl
import numpy as np
import logging


def import_data(path: str):
    return pl.read_parquet(path)

def add_text_length(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("text").str.len_chars().alias("text_length")
    )

