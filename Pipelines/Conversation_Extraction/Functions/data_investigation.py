"""Investigate descriptive statistics for cleaned ACT data."""

## IMPORTS ##
# Standard library
import logging
from math import ceil
from pathlib import Path

# External
import polars as pl
from docx import Document
## _______ ##

## HELPER FUNCTIONS ##
def _shortest_after_removing_bottom_percent(
    df: pl.DataFrame,
    bottom_percent: float,
    logger: logging.Logger,
) -> int | None:
    values = (
        df
        .select(
            pl.col("text_words")
            .drop_nulls()
            .sort()
        )
        .to_series()
        .to_list()
    )

    if not values:
        logger.warning("No wordcount values found.")
        return None

    n_to_remove = ceil(len(values) * (bottom_percent / 100))

    if n_to_remove >= len(values):
        logger.warning(
            f"Bottom percent removal removes entire dataset: "
            f"n_values={len(values)}, "
            f"n_to_remove={n_to_remove}, "
            f"bottom_percent={bottom_percent}"
        )
        return None

    return values[n_to_remove]

def _calculate_wordcount_stats_for_df(
    df: pl.DataFrame,
    bottom_percent: float,
    logger: logging.Logger,
) -> dict:
    logger.debug(f"Calculating wordcount stats for {df.height} rows.")

    shortest_after_trim = _shortest_after_removing_bottom_percent(
        df=df,
        bottom_percent=bottom_percent,
        logger=logger,
    )

    stats = (
        df
        .select(
            pl.len().alias("n_texts"),
            pl.col("text_words").min().alias("min_wordcount"),
            pl.col("text_words").max().alias("max_wordcount"),
            pl.col("text_words").mean().round(1).alias("avg_wordcount"),
            pl.col("text_words").median().alias("med_wordcount"),
        )
        .to_dicts()[0]
    )

    stats[
        f"shortest_after_removing_bottom_{bottom_percent:g}_percent"
    ] = shortest_after_trim

    return stats

def _calculate_grouped_wordcount_stats(
    df: pl.DataFrame,
    group_cols: list[str] | None,
    bottom_percent: float,
    logger: logging.Logger,
) -> pl.DataFrame:
    if group_cols is None:
        logger.info("Calculating wordcount stats for entire dataset.")

        stats = _calculate_wordcount_stats_for_df(
            df=df,
            bottom_percent=bottom_percent,
            logger=logger,
        )

        return pl.DataFrame([{"group": "Entire dataset", **stats}])

    logger.info(
        f"Calculating wordcount stats grouped by: {', '.join(group_cols)}"
    )

    rows = []

    for group_key, group_df in df.group_by(group_cols, maintain_order=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        group_values = dict(zip(group_cols, group_key))

        logger.debug(
            f"Calculating stats for group {group_values} with {group_df.height} rows."
        )

        stats = _calculate_wordcount_stats_for_df(
            df=group_df,
            bottom_percent=bottom_percent,
            logger=logger,
        )

        rows.append({**group_values, **stats})

    return pl.DataFrame(rows)

def _create_wordcount_tables(
    df: pl.DataFrame,
    bottom_percent: float,
    logger: logging.Logger,
) -> dict[str, pl.DataFrame]:
    logger.info("Creating wordcount statistics tables.")

    tables = {
        "entire_dataset": _calculate_grouped_wordcount_stats(
            df=df,
            group_cols=None,
            bottom_percent=bottom_percent,
            logger=logger,
        ),
        "by_gender": _calculate_grouped_wordcount_stats(
            df=df,
            group_cols=["gender"],
            bottom_percent=bottom_percent,
            logger=logger,
        ),
        "by_age_group": _calculate_grouped_wordcount_stats(
            df=df,
            group_cols=["age_group"],
            bottom_percent=bottom_percent,
            logger=logger,
        ),
    }

    for gender in ["female", "male", "other"]:
        gender_df = df.filter(pl.col("gender") == gender)

        if gender_df.is_empty():
            logger.warning(f"No rows found for gender: {gender}")
            continue

        tables[f"by_age_group_for_{gender}"] = _calculate_grouped_wordcount_stats(
            df=gender_df,
            group_cols=["age_group"],
            bottom_percent=bottom_percent,
            logger=logger,
        )

    return tables
    
def _prettify_column_name(column_name: str) -> str:
    replacements = {
        "group": "Group",
        "gender": "Gender",
        "age_group": "Age group",
        "n_texts": "Number of texts",
        "min_wordcount": "Minimum wordcount",
        "max_wordcount": "Maximum wordcount",
        "avg_wordcount": "Average wordcount",
        "med_wordcount": "Median wordcount",
    }

    if column_name in replacements:
        return replacements[column_name]

    if column_name.startswith("shortest_after_removing_bottom_"):
        percent = (
            column_name
            .replace("shortest_after_removing_bottom_", "")
            .replace("_percent", "")
        )
        return f"Shortest wordcount after removing bottom {percent}%"

    return column_name.replace("_", " ").title()

def _add_polars_table_to_docx(
    doc: Document,
    df: pl.DataFrame,
    title: str,
    logger: logging.Logger,
) -> None:
    logger.info(f"Adding table to docx section: {title}")

    doc.add_heading(title.replace("_", " ").title(), level=2)

    table = doc.add_table(rows=1, cols=len(df.columns))
    table.style = "Table Grid"

    header_cells = table.rows[0].cells
    for i, column_name in enumerate(df.columns):
        header_cells[i].text = _prettify_column_name(column_name)

    for row in df.iter_rows(named=True):
        cells = table.add_row().cells

        for i, column_name in enumerate(df.columns):
            value = row[column_name]
            cells[i].text = "" if value is None else str(value)

    doc.add_paragraph()

def _save_wordcount_tables(
    tables: dict[str, pl.DataFrame],
    output_dir_data: str,
    logger: logging.Logger,
) -> None:
    output_path = Path(output_dir_data)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating docx report at {output_path}")

    doc = Document()
    doc.add_heading("Wordcount Statistics", level=1)

    for table_name, table_df in tables.items():
        _add_polars_table_to_docx(
            doc=doc,
            df=table_df,
            title=table_name,
            logger=logger,
        )

    doc.save(output_path)

    logger.info(f"Saved docx report to {output_path}")
## ________________ ##

## MAIN FUNCTIONALITY ##
def investigate_data_stats(
    input_dir_data: str,
    output_dir_data: str,
    logger: logging.Logger,
    bottom_percent: float,
) -> None:
    logger.info(f"Reading input data from {input_dir_data}")

    df = pl.read_parquet(input_dir_data)

    logger.info(f"Loaded dataset with {df.height} rows and {df.width} columns.")

    tables = _create_wordcount_tables(
        df=df,
        bottom_percent=bottom_percent,
        logger=logger,
    )

    _save_wordcount_tables(
        tables=tables,
        output_dir_data=output_dir_data,
        logger=logger,
    )

    logger.info("Finished creating wordcount statistics tables.")
## __________________ ##