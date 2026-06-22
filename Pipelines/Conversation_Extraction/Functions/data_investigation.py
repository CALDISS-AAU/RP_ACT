"""Investigate descriptive statistics for cleaned ACT data."""

## IMPORTS ##
# Standard library
import logging
from math import ceil
from pathlib import Path

# External
import matplotlib.pyplot as plt
import polars as pl
from docx import Document
from docx.shared import Inches
## _______ ##


## HELPER FUNCTIONS ##
def _shortest_after_removing_bottom_percent(
    df: pl.DataFrame,
    bottom_percent: float,
    logger: logging.Logger,
) -> int | None:
    values = (
        df
        .select(pl.col("text_words").drop_nulls().sort())
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

    stats[f"shortest_after_removing_bottom_{bottom_percent:g}_percent"] = (
        shortest_after_trim
    )

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

    logger.info(f"Calculating wordcount stats grouped by: {', '.join(group_cols)}")

    df = df.filter(
        pl.all_horizontal([
            pl.col(col).is_not_null()
            for col in group_cols
        ])
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

def _create_report_sections(
    df: pl.DataFrame,
    bottom_percent: float,
    logger: logging.Logger,
) -> dict[str, dict]:
    logger.info("Creating report sections.")

    sections = {
        "entire_dataset": {
            "title": "Entire dataset",
            "source_df": df,
            "stats_df": _calculate_grouped_wordcount_stats(
                df=df,
                group_cols=None,
                bottom_percent=bottom_percent,
                logger=logger,
            ),
            "chart_group_col": None,
        },
        "by_gender": {
            "title": "By gender",
            "source_df": df,
            "stats_df": _calculate_grouped_wordcount_stats(
                df=df,
                group_cols=["gender"],
                bottom_percent=bottom_percent,
                logger=logger,
            ),
            "chart_group_col": "gender",
        },
        "by_age_group": {
            "title": "By age group",
            "source_df": df,
            "stats_df": _calculate_grouped_wordcount_stats(
                df=df,
                group_cols=["age_group"],
                bottom_percent=bottom_percent,
                logger=logger,
            ),
            "chart_group_col": "age_group",
        },
    }

    for gender in ["female", "male", "other"]:
        gender_df = df.filter(pl.col("gender") == gender)

        if gender_df.is_empty():
            logger.warning(f"No rows found for gender: {gender}")
            continue

        section_name = f"by_age_group_for_{gender}"

        sections[section_name] = {
            "title": f"By age group for {gender}",
            "source_df": gender_df,
            "stats_df": _calculate_grouped_wordcount_stats(
                df=gender_df,
                group_cols=["age_group"],
                bottom_percent=bottom_percent,
                logger=logger,
            ),
            "chart_group_col": "age_group",
        }

    return sections


def _add_wordcount_interval(
    df: pl.DataFrame,
    wordcount_interval_size: int,
    wordcount_stop_at: int,
) -> pl.DataFrame:
    return (
        df
        .with_columns(
            pl.when(pl.col("text_words") >= wordcount_stop_at)
            .then(wordcount_stop_at)
            .otherwise(
                (pl.col("text_words") // wordcount_interval_size)
                * wordcount_interval_size
            )
            .alias("interval_start")
        )
        .with_columns(
            pl.when(pl.col("interval_start") == wordcount_stop_at)
            .then(pl.lit(f"{wordcount_stop_at}+"))
            .otherwise(
                pl.col("interval_start").cast(pl.Utf8)
                + "-"
                + (
                    pl.col("interval_start") + wordcount_interval_size - 1
                ).cast(pl.Utf8)
            )
            .alias("wordcount_interval")
        )
    )


def _calculate_wordcount_distribution(
    df: pl.DataFrame,
    group_col: str | None,
    wordcount_interval_size: int,
    wordcount_stop_at: int,
    logger: logging.Logger,
) -> pl.DataFrame:
    logger.info(
        f"Calculating wordcount distribution. "
        f"group_col={group_col}, "
        f"wordcount_interval_size={wordcount_interval_size}"
    )

    df = df.filter(pl.col("text_words").is_not_null())

    if group_col is not None:
        df = df.filter(pl.col(group_col).is_not_null())

    df = _add_wordcount_interval(
        df=df,
        wordcount_interval_size=wordcount_interval_size,
        wordcount_stop_at=wordcount_stop_at,
    )

    if group_col is None:
        return (
            df
            .group_by(["interval_start", "wordcount_interval"])
            .agg(pl.len().alias("n_texts"))
            .sort("interval_start")
            .drop("interval_start")
        )

    return (
        df
        .group_by([group_col, "interval_start", "wordcount_interval"])
        .agg(pl.len().alias("n_texts"))
        .sort([group_col, "interval_start"])
        .drop("interval_start")
    )

def _save_wordcount_distribution_chart(
    distribution_df: pl.DataFrame,
    chart_title: str,
    chart_group_col: str | None,
    output_path: Path,
    logger: logging.Logger,
) -> None:
    logger.info(f"Saving chart to {output_path}")

    plt.figure(figsize=(10, max(5, distribution_df.height * 0.35)))

    if chart_group_col is None:
        plt.barh(
            distribution_df["wordcount_interval"].to_list(),
            distribution_df["n_texts"].to_list(),
        )

    else:
        pivot_df = (
            distribution_df
            .pivot(
                values="n_texts",
                index="wordcount_interval",
                columns=chart_group_col,
                aggregate_function="sum",
            )
            .fill_null(0)
        )

        y_labels = pivot_df["wordcount_interval"].to_list()
        group_columns = [
            col for col in pivot_df.columns
            if col != "wordcount_interval"
        ]

        y_positions = list(range(len(y_labels)))
        bar_height = 0.8 / len(group_columns)

        for i, group_value in enumerate(group_columns):
            positions = [
                y + (i * bar_height) - (0.4 - bar_height / 2)
                for y in y_positions
            ]

            plt.barh(
                positions,
                pivot_df[group_value].to_list(),
                height=bar_height,
                label=str(group_value),
            )

        plt.yticks(y_positions, y_labels)
        plt.legend(title=chart_group_col.replace("_", " ").title())

    plt.title(chart_title)
    plt.xlabel("Number of texts")
    plt.ylabel("Wordcount interval")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

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

    doc.add_heading(title, level=2)

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


def _add_chart_to_docx(
    doc: Document,
    chart_path: Path,
    logger: logging.Logger,
) -> None:
    logger.info(f"Adding chart to docx: {chart_path}")

    doc.add_picture(str(chart_path), width=Inches(6.5))
    doc.add_paragraph()


def _save_report_to_docx(
    sections: dict[str, dict],
    output_dir_data: str,
    output_dir_pngs: str,
    wordcount_interval_size: int,
    wordcount_stop_at: int,
    logger: logging.Logger,
) -> None:
    output_docx_path = Path(output_dir_data)
    output_docx_path.parent.mkdir(parents=True, exist_ok=True)

    output_png_dir = Path(output_dir_pngs)
    output_png_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating docx report at {output_docx_path}")

    doc = Document()
    doc.add_heading("Wordcount Statistics", level=1)

    for section_name, section in sections.items():
        title = section["title"]
        source_df = section["source_df"]
        stats_df = section["stats_df"]
        chart_group_col = section["chart_group_col"]

        _add_polars_table_to_docx(
            doc=doc,
            df=stats_df,
            title=title,
            logger=logger,
        )

        distribution_df = _calculate_wordcount_distribution(
            df=source_df,
            group_col=chart_group_col,
            wordcount_interval_size=wordcount_interval_size,
            wordcount_stop_at=wordcount_stop_at,
            logger=logger,
        )

        chart_path = output_png_dir / f"{section_name}_wordcount_distribution.png"

        _save_wordcount_distribution_chart(
            distribution_df=distribution_df,
            chart_title=f"{title}: wordcount distribution",
            chart_group_col=chart_group_col,
            output_path=chart_path,
            logger=logger,
        )

        _add_chart_to_docx(
            doc=doc,
            chart_path=chart_path,
            logger=logger,
        )

    doc.save(output_docx_path)

    logger.info(f"Saved docx report to {output_docx_path}")
## ________________ ##


## MAIN FUNCTIONALITY ##
def investigate_data_stats(
    input_dir_data: str,
    output_dir_data: str,
    output_dir_pngs: str,
    logger: logging.Logger,
    bottom_percent: float,
    wordcount_interval_size: int,
    wordcount_stop_at: int,
) -> None:
    logger.info(f"Reading input data from {input_dir_data}")

    df = pl.read_parquet(input_dir_data)

    logger.info(f"Loaded dataset with {df.height} rows and {df.width} columns.")

    sections = _create_report_sections(
        df=df,
        bottom_percent=bottom_percent,
        logger=logger,
    )

    _save_report_to_docx(
        sections=sections,
        output_dir_data=output_dir_data,
        output_dir_pngs=output_dir_pngs,
        wordcount_interval_size=wordcount_interval_size,
        wordcount_stop_at=wordcount_stop_at,
        logger=logger,
    )

    logger.info("Finished creating wordcount statistics report.")
## __________________ ##