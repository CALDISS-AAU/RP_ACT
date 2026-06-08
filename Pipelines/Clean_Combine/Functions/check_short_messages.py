import pandas as pd

# CHANGE THESE
INPUT_FILE = "/work/CALDISS-projects/ACT/Pipelines/Clean_Combine/Data/combined_data.jsonl"
OUTPUT_FILE = "/work/CALDISS-projects/ACT/Pipelines/Clean_Combine/Data/short_messages.txt"
META_OUTPUT_FILE = "/work/CALDISS-projects/ACT/Pipelines/Clean_Combine/Data/meta_messages.txt"
COLUMN_NAME = "msg_tekst"
MAX_LENGTH = 25

df = pd.read_json(INPUT_FILE, lines=True)

df["msg_length"] = (
    df["msg_tekst"]
    .fillna("")
    .astype(str)
    .str.len()
)

age_summary = (
    df.groupby("requester_age_name")["msg_length"]
    .agg(
        count="count",
        mean="mean",
        median="median",
        std="std",
        pct_short=lambda x: (x < MAX_LENGTH).mean() * 100,
    )
    .round(2)
    .sort_index()
)

print(age_summary)

meta_messages = df[
    df["msg_tekst"]
    .fillna("")
    .str.contains(
        r"chat",
        case=False,
        regex=True,
        na=False,
    )
]

meta_messages[["conversation_code", "msg_tekst"]].to_csv(
    META_OUTPUT_FILE,
    index=False,
)