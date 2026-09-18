"""Print all unique contact reasons from a parquet file."""

import polars as pl


INPUT_FILE = "/work/ACT/Data/clean_combine/conversation_data.parquet"


def main() -> None:
    df = pl.read_parquet(INPUT_FILE)

    unique_contact_reasons = (
        df
        .select("contact_reasons")
        .explode("contact_reasons")
        .drop_nulls()
        .unique()
        .sort("contact_reasons")
        .get_column("contact_reasons")
        .to_list()
    )

    print(f"\nFound {len(unique_contact_reasons)} unique contact reasons:\n")

    for contact_reason in unique_contact_reasons:
        print(contact_reason)


if __name__ == "__main__":
    main()