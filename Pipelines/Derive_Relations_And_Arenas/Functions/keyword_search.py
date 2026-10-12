
"""Processing functions for the Derive_Relations_And_Arenas pipeline."""

from collections import Counter
from collections.abc import Iterable
from pathlib import Path
import re

import pandas as pd
import polars as pl
from spacy.lang.da import Danish


def _build_stop_words(
    stopwords_extension: Iterable[str],
    maybe_stopwords: Iterable[str],
) -> set[str]:
    nlp = Danish()
    stop_words = set(nlp.Defaults.stop_words)
    print(stop_words)
    stop_words.update(stopwords_extension)
    stop_words.update(maybe_stopwords)
    return stop_words


def _count_used_words(
    input_df: pl.DataFrame,
    *,
    stop_words: set[str],
    min_word_count: int,
    min_text_count: int,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    word_counts = Counter()
    text_counts = Counter()

    for text in input_df["msg_tekst"]:
        all_words = re.findall(r"\b[\w']+\b", text.lower())
        new_words = [word for word in all_words if word not in stop_words]
        word_counts.update(new_words)
        text_counts.update(set(new_words))

    filtered_words = Counter({
        word: count
        for word, count in word_counts.items()
        if count > min_word_count
    })
    filtered_texts = Counter({
        word: count
        for word, count in text_counts.items()
        if count > min_text_count
    })
    return filtered_words.most_common(), filtered_texts.most_common()


def _combine_to_xlsx(
    counts: dict[str, tuple[list[tuple[str, int]], list[tuple[str, int]]]],
    output_path: str | Path,
) -> None:
    columns = {}
    for name, (words, conversations) in counts.items():

        columns[(name, "Words")] = pd.Series(
            [word for word, _ in words], dtype="string"
        )
        columns[(name, "Conversations")] = pd.Series(
            [word for word, _ in conversations], dtype="string"
        )

    table = pd.DataFrame(columns)
    table.columns = range(table.shape[1])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        table.to_excel(
            writer,
            sheet_name="Counts",
            startrow=2,
            header=False,
            index=False,
        )
        workbook = writer.book
        sheet = writer.sheets["Counts"]
        group_header = workbook.add_format({
            "bold": True, "align": "center", "right": 5
        })
        subheader = workbook.add_format({"bold": True})
        subheader_right = workbook.add_format({"bold": True, "right": 5})
        right_border = workbook.add_format({"right": 5})

        for group_index, name in enumerate(counts):
            left_col = group_index * 2
            right_col = left_col + 1
            sheet.merge_range(0, left_col, 0, right_col, name, group_header)
            sheet.write(1, left_col, "Words", subheader)
            sheet.write(1, right_col, "Conversations", subheader_right)
            sheet.set_column(left_col, right_col, 24)

            # Format every cell on the group's right edge, including blanks.
            for row_index in range(len(table)):
                value = table.iloc[row_index, right_col]
                excel_row = row_index + 2
                if pd.isna(value):
                    sheet.write_blank(excel_row, right_col, None, right_border)
                else:
                    sheet.write_string(
                        excel_row, right_col, value, right_border
                    )

        sheet.freeze_panes(2, 0)


def derive_from_keyword_search(
    *,
    input_path: str | Path,
    output_path: str | Path,
    maybe_stopwords: Iterable[str],
    stopwords_extension: Iterable[str],
    min_word_count: int,
    min_text_count: int,
) -> None:
    """Read conversations, count words by group, and write the Excel workbook."""
    stop_words = _build_stop_words(stopwords_extension, maybe_stopwords)
    df = pl.read_parquet(input_path)
    groups = {
        "All": df,
        "Boys": df.filter(pl.col("gender") == "male"),
        "Girls": df.filter(pl.col("gender") == "female"),
        "Pre-teens": df.filter(pl.col("age_group") == "pre teen"),
        "Early-teens": df.filter(pl.col("age_group") == "early teen"),
        "Teens": df.filter(pl.col("age_group") == "teen"),
    }
    counts = {
        name: _count_used_words(
            group_df,
            stop_words=stop_words,
            min_word_count=min_word_count,
            min_text_count=min_text_count,
        )
        for name, group_df in groups.items()
    }
    _combine_to_xlsx(counts, output_path)
