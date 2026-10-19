"""Extract Danish phrases and export token annotations and phrase counts."""

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

CSV_COLUMNS = [
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
]


def __normalise_phrase(tokens: list[Token]) -> str:
    """Build a lowercase phrase from token lemmas."""
    return " ".join(
        (token.lemma_ or token.text).casefold()
        for token in tokens
    )


def _get_phrase_tokens(head: Token) -> list[Token]:
    """Collect a head and its directly attached preceding modifiers."""
    context = [
        child
        for child in head.children
        if child.i < head.i
        and (child.pos_, child.dep_) in CONTEXT_RULES
    ]
    return sorted([*context, head], key=lambda token: token.i)


def _analyse_texts(texts, conversation_codes, nlp):
    """Build CSV rows and collect (primary word, phrase) occurrences."""
    token_rows = []
    occurrences_by_conversation = {}

    for text_index, text in enumerate(texts):
        if text is None or not text.strip():
            continue

        doc = nlp(text)
        conversation_code = str(conversation_codes[text_index])
        occurrences = occurrences_by_conversation.setdefault(
            conversation_code, []
        )

        phrases_by_head = {}
        primary_words_by_head = {}
        included_indices = set()

        for head in doc:
            if not (head.pos_ == "NOUN" or head.dep_ == "nmod"):
                continue

            phrase_tokens = _get_phrase_tokens(head)
            phrase = " ".join(token.text for token in phrase_tokens)
            primary_word = (head.lemma_ or head.text).casefold()

            phrases_by_head[head.i] = phrase
            primary_words_by_head[head.i] = primary_word
            included_indices.update(token.i for token in phrase_tokens)

            normalised_phrase = __normalise_phrase(phrase_tokens)
            occurrences.append((primary_word, normalised_phrase))

        print(f"Tekst: {text}")
        print(f"Relevante udtryk: {list(phrases_by_head.values())}")

        for token in doc:
            selected_by_current_rule = (
                token.pos_ == "NOUN"
                and token.dep_ in CURRENT_NOUN_DEPENDENCIES
            )
            token_rows.append([
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

    return token_rows, occurrences_by_conversation


def _reduce_to_phrases_on_conversation_level(occurrences_by_conversation):
    """Group each conversation's phrases by their primary word."""
    phrases_by_conversation = {}

    for conversation_code, occurrences in occurrences_by_conversation.items():
        conversation_phrases = {}

        for primary_word, phrase in occurrences:
            conversation_phrases.setdefault(primary_word, []).append(phrase)

        phrases_by_conversation[conversation_code] = conversation_phrases

    return phrases_by_conversation


def _summarise_phrases_across_all_conversations(occurrences_by_conversation):
    """Count phrases globally and sort words and phrases by frequency."""
    phrase_counts_by_word = {}

    for occurrences in occurrences_by_conversation.values():
        for primary_word, phrase in occurrences:
            counts = phrase_counts_by_word.setdefault(primary_word, Counter())
            counts[phrase] += 1

    sorted_words = sorted(
        phrase_counts_by_word.items(),
        key=lambda item: sum(item[1].values()),
        reverse=True,
    )

    return {
        primary_word: dict(counts.most_common())
        for primary_word, counts in sorted_words
    }


def derive_from_spacy_search(
    input_path: str,
    output_path: str,
) -> None:
    df = pl.read_parquet(input_path)
    df = df[:10]
    df_texts = df["msg_tekst"]
    df_conversation_code = df["conversation_code"]
    nlp = spacy.load("da_core_news_trf")

    output_file = Path(output_path)  # CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    json_file = output_file.parent / "relations_and_arenas_pr_conversation.json"
    summary_file = output_file.parent / "summary.json"

    # Build the three outputs.
    token_rows, occurrences_by_conversation = _analyse_texts(
        df_texts, df_conversation_code, nlp
    )
    phrases_by_conversation = _reduce_to_phrases_on_conversation_level(
        occurrences_by_conversation
    )
    summary = _summarise_phrases_across_all_conversations(
        occurrences_by_conversation
    )

    # Save the outputs.
    with output_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(CSV_COLUMNS)
        writer.writerows(token_rows)

    with json_file.open("w", encoding="utf-8") as file:
        json.dump(phrases_by_conversation, file, ensure_ascii=False, indent=2)

    with summary_file.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(f"CSV gemt: {output_file}")
    print(f"JSON gemt: {json_file}")
    print(f"Summary JSON gemt: {summary_file}")