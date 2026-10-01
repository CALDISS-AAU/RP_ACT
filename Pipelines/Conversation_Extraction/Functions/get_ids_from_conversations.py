"""Finds BV conversation ID for identified sensitive conversations - NOTE: Male Teen 31 was already deleted from the data (AA840612)"""
from pathlib import Path
from docx import Document
import pandas as pd

## STATIC VARIABLES ##
# Directories - internal output
OUTPUT_DIR_PIPE = Path("./Pipelines/Conversation_Extraction/Data")

# Directories - global output
OUTPUT_DIR_BASE = Path("./Output/Conversation_Extraction")
OUTPUT_DIR_STRATIFIED_CONVERSATIONS = (
    OUTPUT_DIR_BASE / "Stratified_Conversations"
)

# Conversations to find
convos_find = {
    "Male": {
        "Early_Teen": [28, 47],
        "Pre_Teen": [3, 6, 24, 26, 39, 47],
        "Teen": [8, 16, 31]
    },
    "Other": {
        "Early_Teen": [4, 5],
        "Pre_Teen": [1],
        "Teen": [3]

    }
}

def _get_conversation_code(file_path: Path) -> str:
    """
    Extract the conversation code from a DOCX file.

    Expected paragraph format:
        Samtale ID : <conversation_code>
    """
    doc = Document(file_path)

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()

        if text.startswith("Samtale ID"):
            _, separator, conversation_code = text.partition(":")

            if separator:
                return conversation_code.strip()

    raise ValueError(
        f"Could not find 'Samtale ID' in document: {file_path}"
    )

def main():
    lookup_dict = {}

    for gender, agecat_dict in convos_find.items():

        for agecat, ids in agecat_dict.items():

            for convo_number in ids:

                convo_id_internal = f"{gender} - {agecat} {convo_number:03d}"
                file_path = OUTPUT_DIR_STRATIFIED_CONVERSATIONS / gender / agecat / f"conversation_{convo_number:03d}.docx"

                if not file_path.exists():
                    print(f"WARNING: File not found: {file_path}")
                    continue

                try:
                    bv_id = _get_conversation_code(file_path)

                except ValueError as exc:
                    print(f"WARNING: {exc}")
                    continue

                lookup_dict[convo_id_internal] = bv_id

    # Convert dictionary to DataFrame.
    lookup_df = (
        pd.DataFrame.from_dict(
            lookup_dict,
            orient="index",
            columns=["conversation_code"],
        )
        .rename_axis("conversation_id_internal")
        .reset_index()
    )

    # Output as csv
    OUTPUT_DIR_PIPE.mkdir(parents=True, exist_ok=True)

    outpath = OUTPUT_DIR_PIPE / "sensitive_convos_ids.csv"
    lookup_df.to_csv(outpath, index=False)

    print(f"Wrote {len(lookup_df)} conversation IDs to: {outpath}")


## CALL OF MAIN FUNCTION ##
if __name__ == "__main__":
    main()