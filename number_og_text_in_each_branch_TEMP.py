"""Print conversation counts for combinations of gender, age group, and text length."""

import polars as pl
from itertools import product


INPUT_DIR_DATA = "/work/RP_ACT/Data/conversation_data.parquet"


def main() -> None:
    df = pl.read_parquet(INPUT_DIR_DATA)

    df = df.with_columns(
        pl.when(pl.col("text_words").is_between(200, 399))
        .then(pl.lit("200-400"))
        .when(pl.col("text_words").is_between(400, 799))
        .then(pl.lit("400-800"))
        .when(pl.col("text_words") >= 800)
        .then(pl.lit("800+"))
        .otherwise(None)
        .alias("text_word_group")
    )

    all_combinations = pl.DataFrame(
        product(
            ["male", "female"],
            ["pre teen", "early teen", "teen"],
            ["200-400", "400-800", "800+"],
        ),
        schema=["gender", "age_group", "text_word_group"],
        orient="row",
    )

    counts = (
        df.filter(
            pl.col("gender").is_in(["male", "female"])
            & pl.col("age_group").is_in(["pre teen", "early teen", "teen"])
            & pl.col("text_word_group").is_not_null()
        )
        .group_by(["gender", "age_group", "text_word_group"])
        .agg(
            pl.col("conversation_code")
            .n_unique()
            .alias("n_conversations")
        )
    )

    summary = (
        all_combinations
        .join(
            counts,
            on=["gender", "age_group", "text_word_group"],
            how="left",
        )
        .with_columns(
            pl.col("n_conversations").fill_null(0)
        )
        .sort(["gender", "age_group", "text_word_group"])
    )

    with pl.Config(tbl_cols=-1, tbl_rows=-1):
        print(summary)


if __name__ == "__main__":
    main()