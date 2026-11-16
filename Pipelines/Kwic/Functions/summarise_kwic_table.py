"""Summarise the most frequent context words for each KWIC keyword."""

## IMPORTS ##
import json
import logging
import re
from collections import Counter
from pathlib import Path

import polars as pl
## _______ ##


## HELPER FUNCTIONS ##
def __extract_words(text: str | None) -> list[str]:
    """Extract lowercase words, including Danish letters."""
    if not text:
        return []

    return re.findall(r"[^\W\d_]+", text.casefold())


def _generate_context_counters(
    kwic_table: pl.DataFrame,
    logger: logging.Logger,
) -> tuple[
    dict[str, dict[str, Counter[str]]],
    dict[str, dict[str, Counter[str]]],
]:
    """Count words before and after each keyword within each category."""
    before_counters: dict[str, dict[str, Counter[str]]] = {}
    after_counters: dict[str, dict[str, Counter[str]]] = {}

    for row in kwic_table.iter_rows(named=True):
        category = row["Category"]
        keyword = row["Keyword"]

        if category not in before_counters:
            before_counters[category] = {}
            after_counters[category] = {}

        if keyword not in before_counters[category]:
            before_counters[category][keyword] = Counter()
            after_counters[category][keyword] = Counter()

        before_counters[category][keyword].update(
            __extract_words(row["n words before kw"])
        )
        after_counters[category][keyword].update(
            __extract_words(row["n words after kw"])
        )

    logger.info(
        "Generated context counters for %s KWIC rows",
        kwic_table.height,
    )

    return before_counters, after_counters


def _summarise_context_counters(
    before_counters: dict[str, dict[str, Counter[str]]],
    after_counters: dict[str, dict[str, Counter[str]]],
    top_m: int,
    logger: logging.Logger,
) -> dict[str, dict[str, dict[str, int]]]:
    """Combine context counts and select the top m words per keyword."""
    summary: dict[str, dict[str, dict[str, int]]] = {}

    for category, keywords in before_counters.items():
        summary[category] = {}

        for keyword, before_counter in keywords.items():
            after_counter = after_counters[category][keyword]
            total_counter = before_counter + after_counter

            summary[category][keyword] = dict(
                total_counter.most_common(top_m)
            )

    logger.info(
        "Summarised context words for %s category-keyword combinations",
        sum(len(keywords) for keywords in summary.values()),
    )

    return summary
## ________________ ##


## MAIN FUNCTIONALITY ##
def summarise_kwic_table(
    input_dir: str,
    output_dir: str,
    top_m: int,
    logger: logging.Logger,
) -> None:
    """Summarise KWIC context words and save the result as JSON."""
    if top_m < 0:
        raise ValueError("top_m must be non-negative.")

    logger.info("Beginning KWIC context summary")

    # Read input data.
    kwic_table = pl.read_csv(input_dir, separator=";")
    logger.info("Read %s KWIC rows from %s", kwic_table.height, input_dir)

    # Generate context counters.
    before_counters, after_counters = _generate_context_counters(
        kwic_table=kwic_table,
        logger=logger,
    )

    # Summarise context counters.
    summary = _summarise_context_counters(
        before_counters=before_counters,
        after_counters=after_counters,
        top_m=top_m,
        logger=logger,
    )

    # Save the summary.
    output_path = Path(output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            summary,
            output_file,
            ensure_ascii=False,
            indent=4,
        )

    logger.info("Saved KWIC context summary to %s", output_path)
## ____________________ ##