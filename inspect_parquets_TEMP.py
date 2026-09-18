import polars as pl

parquet_dir_1 = "/work/RP_ACT/Data/clean_combine/message_data.parquet"
parquet_dir_2 = "/work/RP_ACT/Data/clean_combine/collapsed_messages_data.parquet"
parquet_dir_3 = "/work/RP_ACT/Data/clean_combine/conversation_data.parquet"

df1 = pl.read_parquet(parquet_dir_1)
df2 = pl.read_parquet(parquet_dir_2)
df3 = pl.read_parquet(parquet_dir_3)

def _find_random_conversation_code(
    dataframe: pl.DataFrame,
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

    return conversation_code

conversation = _find_random_conversation_code(df1, 5)

with pl.Config(tbl_cols=-1, fmt_str_lengths=500):
    # print("df1: message_data")
    # print(df1.filter(pl.col("conversation_code") == conversation).head())

    # print("\ndf2: collapsed_messages_data")
    # print(df2.filter(pl.col("conversation_code") == conversation).head())

    print("\ndf3: conversation_data")
    print(df3["gender", "age_group", "msg_tekst"].head())