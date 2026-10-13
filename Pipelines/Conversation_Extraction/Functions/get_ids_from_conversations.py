"""Finds BV conversation ID for identified sensitive conversations - NOTE: Male Teen 31 was already deleted from the data (AA840612)"""
from pathlib import Path
from docx import Document
from docx.opc.exceptions import PackageNotFoundError
import pandas as pd
import re
import json

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


def _sanitize_folder_name(value: str) -> str:
    """Convert a value into a safe folder name."""
    value = value.strip().title()
    value = re.sub(r"[^a-zA-Z0-9æøåÆØÅ_-]+", "_", value)
    value = re.sub(r"_+", "_", value)

    return value.strip("_")


def create_conversation_code_mapping(
    sample_plan: dict[str, dict[str, int]],
    input_dir: str,
    output_dir: str,
):
    mapping_dict = {}

    for gender in sample_plan:
        mapping_dict[gender] = {}
        sanitized_gender = _sanitize_folder_name(gender)

        for agegroup in sample_plan[gender]:
            mapping_dict[gender][agegroup] = {}
            sanitized_agegroup = _sanitize_folder_name(agegroup)

            range_list = [
                f"{number:03d}"
                for number in range(1, sample_plan[gender][agegroup] + 1)
            ]

            for number in range_list:
                file_name = f"conversation_{number}"
                file_path = (
                    f"{input_dir}/{sanitized_gender}/"
                    f"{sanitized_agegroup}/{file_name}.docx"
                )
                try:
                    convo_ID = _get_conversation_code(file_path)
                except (FileNotFoundError, PackageNotFoundError) as error:
                    print(f"Could not open file: {file_path} ({error})")
                    continue

                mapping_dict[gender][agegroup][file_name] = convo_ID

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with (output_path / "conversation_code_mapping.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(mapping_dict, file, indent=4, ensure_ascii=False)
