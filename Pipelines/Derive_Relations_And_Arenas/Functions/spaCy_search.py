"""Extract Danish tokens and export their annotations to CSV."""

import csv
from pathlib import Path

import polars as pl
import spacy


def derive_from_spacy_search(
    input_path: str,
    output_path: str,
) -> None:
    df_texts = pl.read_parquet(input_path)["msg_tekst"]
    nlp = spacy.load("da_core_news_trf")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open(
        "w", encoding="utf-8-sig", newline=""
    ) as csv_file:
        writer = csv.writer(csv_file, delimiter=";")
        writer.writerow([
            "Text_index",
            "Token_index",
            "Word",
            "Lemma",
            "POS",
            "Dependency",
            "Head",
            "Selected_by_current_rule",
        ])

        for text_index, text in enumerate(df_texts[:5]):
            if text is None or not text.strip():
                continue

            doc = nlp(text)

            relevant_tokens = [
                token
                for token in doc
                if token.pos_ == "NOUN"
                and token.dep_ in {
                    "nsubj", "nmod", "list", "obl", "dep", "obj"
                }
            ]
            selected_indices = {token.i for token in relevant_tokens}

            print(f"Tekst: {text}")
            print(
                f"Relevante ord: "
                f"{[token.text for token in relevant_tokens]}"
            )

            for token in doc:
                writer.writerow([
                    text_index,
                    token.i,
                    token.text,
                    token.lemma_,
                    token.pos_,
                    token.dep_,
                    token.head.text,
                    token.i in selected_indices,
                ])

    print(f"CSV gemt: {output_file}")