"""Example helper functions for the pipeline."""

## IMPORTS ##
import csv
import logging
from collections.abc import Iterator
from pathlib import Path

import polars as pl
import spacy
from spacy.language import Language
from spacy.tokens import Doc
## _______ ##


## HELPER FUNCTIONS ##
def _find_top_n_keywords(
    input_dir: str,
    top_n: int,
    logger:logging.Logger
) -> list[str]:
    """Derives top n keywords and returns them."""
    logger.info(f"Began finding top {top_n} keywords")
    df = pl.read_csv(input_dir, separator=";")
    logger.info(f"Read the {input_dir} file")
    keywords = (
        df
        .filter(pl.col("Total__rank") <= top_n)
        .get_column("primary_word")
        .to_list()
    )
    logger.info(f"Derived and returns the following list of keywords: {keywords}")
    return keywords


def __find_keyword_matches(
    doc: Doc,
    keywords: list[str],
) -> list[dict]:
    """Find keywords and record their original forms and token positions."""
    matches = []

    for token in doc:
        lemma = (token.lemma_ or token.text).casefold()

        if lemma in keywords:
            matches.append({
                "keyword": lemma,
                "word_in_text": token.text,
                "token_index": token.i,
            })

    return matches


def __extract_keyword_context(
    doc: Doc,
    token_index: int,
    window_size: int,
) -> tuple[str, str]:
    """Extract up to n words on each side within the keyword's sentence."""
    if window_size < 0:
        raise ValueError("window_size must be zero or greater.")

    if window_size == 0:
        return "", ""

    keyword_token = doc[token_index]
    sentence = keyword_token.sent

    words_before = [
        token for token in sentence
        if token.i < token_index
        and any(character.isalnum() for character in token.text)
    ]
    words_after = [
        token for token in sentence
        if token.i > token_index
        and any(character.isalnum() for character in token.text)
    ]

    # Start at the nth preceding word, or the sentence beginning.
    if len(words_before) > window_size:
        context_start = words_before[-window_size].idx
    else:
        context_start = sentence.start_char

    # Stop before the next excluded word, or at the sentence end.
    if len(words_after) > window_size:
        context_end = words_after[window_size].idx
    else:
        context_end = sentence.end_char

    keyword_start = keyword_token.idx
    keyword_end = keyword_start + len(keyword_token.text)

    before = doc.text[context_start:keyword_start].strip()
    after = doc.text[keyword_end:context_end].strip()

    return before, after


## ________________ ##

## MAIN FUNCTIONALITY ##
def generate_kwic_table(
    input_dir_data: str,
    input_dir_arenas: str,
    input_dir_relations: str,
    output_dir: str,
    words_before_and_after_keyword: int,
    top_relevant_arenas: int,
    top_relevant_relations: int,
    logger: logging.Logger,
) -> None:
    """Generates KWIC table."""

    arenas = _find_top_n_keywords(
        input_dir=input_dir_arenas,
        top_n=top_relevant_arenas,
        logger=logger
    )

    relations = _find_top_n_keywords(
        input_dir=input_dir_relations,
        top_n=top_relevant_relations,
        logger=logger
    )

    df = pl.read_parquet(input_dir_data)
    KWICs = []

    for conversation in df["msg_tekst"]:
        doc = nlp(conversation)
        matches = __find_keyword_matches(doc, keywords)

        for match in matches:
            before, after = __extract_keyword_context(
                doc=doc,
                token_index=match["token_index"],
                window_size=window_size,
            )

            KWICs.append({
                "Keyword": match["keyword"],
                "n words before kw": before,
                "kw in text": match["word_in_text"],
                "n words after kw": after,
            })