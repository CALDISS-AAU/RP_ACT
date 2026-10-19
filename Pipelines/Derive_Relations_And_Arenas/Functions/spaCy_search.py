"""Extract Danish phrases and export annotations and nested phrase counts."""

import csv
import json
from collections import Counter
from pathlib import Path

import polars as pl
import spacy
from spacy.tokens import Token


CURRENT_NOUN_DEPENDENCIES = {
    "nsubj", "nmod", "list", "obl", "dep", "obj"
}

CONTEXT_RULES = {
    ("DET", "det"),
    ("ADJ", "amod"),
    ("ADP", "case"),
}


def _get_phrase_tokens(head: Token) -> list[Token]:
    """Collect a candidate and its directly attached preceding modifiers."""
    context = [
        child
        for child in head.children
        if child.i < head.i
        and (child.pos_, child.dep_) in CONTEXT_RULES
    ]

    return sorted([*context, head], key=lambda token: token.i)


def derive_from_spacy_search(
    input_path: str,
    output_path: str,
) -> None:
    df = pl.read_parquet(input_path)
    df_texts = df["msg_tekst"]
    df_conversation_code = df["conversation_code"]
    nlp = spacy.load("da_core_news_trf")

    output_file = Path(output_path) # CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    json_file = output_file.parent / "relations_and_arenas_pr_conversation.json" # JSON / covo
    summary_file = output_file.parent / "summary.json" # JSON total

    phrases_by_conversation: dict[str, dict[str, list[str]]] = {}
    phrase_counts_by_word: dict[str, Counter[str]] = {}

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
            "Included_in_phrase",
            "Extracted_phrase",
            "Primary_word",
        ])

        for text_index, text in enumerate(df_texts):
            if text is None or not text.strip():
                continue

            doc = nlp(text)

            conversation_code = str(df_conversation_code[text_index])
            conversation_phrases = phrases_by_conversation.setdefault(
                conversation_code, {}
            )

            phrases_by_head: dict[int, str] = {}
            primary_words_by_head: dict[int, str] = {}
            included_indices: set[int] = set()

            for head in doc:
                if not (head.pos_ == "NOUN" or head.dep_ == "nmod"):
                    continue

                phrase_tokens = _get_phrase_tokens(head)
                phrase = " ".join(
                    token.text for token in phrase_tokens
                )
                primary_word = (head.lemma_ or head.text).casefold()

                phrases_by_head[head.i] = phrase
                primary_words_by_head[head.i] = primary_word
                included_indices.update(
                    token.i for token in phrase_tokens
                )

                conversation_phrases.setdefault(
                    primary_word, []
                ).append(phrase)

                phrase_counts_by_word.setdefault(
                    primary_word, Counter()
                ).update([phrase])

            print(f"Tekst: {text}")
            print(
                f"Relevante udtryk: "
                f"{list(phrases_by_head.values())}"
            )

            for token in doc:
                selected_by_current_rule = (
                    token.pos_ == "NOUN"
                    and token.dep_ in CURRENT_NOUN_DEPENDENCIES
                )

                writer.writerow([
                    text_index,
                    token.i,
                    token.text,
                    token.lemma_,
                    token.pos_,
                    token.dep_,
                    token.head.text,
                    selected_by_current_rule,
                    token.i in included_indices,
                    phrases_by_head.get(token.i, ""),
                    primary_words_by_head.get(token.i, ""),
                ])

    with json_file.open("w", encoding="utf-8") as file:
        json.dump(
            phrases_by_conversation,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # Order primary words by total mentions, then phrases by frequency.
    summary = {
        primary_word: dict(counts.most_common())
        for primary_word, counts in sorted(
            phrase_counts_by_word.items(),
            key=lambda item: sum(item[1].values()),
            reverse=True,
        )
    }

    with summary_file.open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"CSV gemt: {output_file}")
    print(f"JSON gemt: {json_file}")
    print(f"Summary JSON gemt: {summary_file}")