"""Example helper functions for the pipeline."""

## IMPORTS ##
import csv
import logging
from pathlib import Path

import polars as pl
import spacy
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
    """Generate one KWIC table with separate arena and relation categories."""
    if words_before_and_after_keyword < 0:
        raise ValueError("words_before_and_after_keyword must be non-negative.")

    logger.info("Beginning KWIC table generation")

    arenas = _find_top_n_keywords(
        input_dir=input_dir_arenas,
        top_n=top_relevant_arenas,
        logger=logger,
    )
    relations = _find_top_n_keywords(
        input_dir=input_dir_relations,
        top_n=top_relevant_relations,
        logger=logger,
    )

    keywords_by_category = {
        "arenas": arenas,
        "relations": relations,
    }

    logger.info("Loading da_core_news_trf")
    nlp = spacy.load("da_core_news_trf")

    logger.info("Reading conversations from %s", input_dir_data)
    df = pl.read_parquet(input_dir_data, columns=["msg_tekst"])
    kwic_rows = []

    for conversation_number, conversation in enumerate(
        df["msg_tekst"], start=1
    ):
        if conversation is None or not conversation.strip():
            continue

        # Process each conversation once, then search both categories.
        doc = nlp(conversation)

        for category, keywords in keywords_by_category.items():
            matches = __find_keyword_matches(doc, keywords)

            for match in matches:
                before, after = __extract_keyword_context(
                    doc=doc,
                    token_index=match["token_index"],
                    window_size=words_before_and_after_keyword,
                )

                kwic_rows.append({
                    "Category": category,
                    "Keyword": match["keyword"],
                    "n words before kw": before,
                    "kw in text": match["word_in_text"],
                    "n words after kw": after,
                })

        if conversation_number % 100 == 0:
            logger.info(
                "Reached conversation %s/%s; collected %s KWIC rows",
                conversation_number,
                df.height,
                len(kwic_rows),
            )

    # An explicit schema preserves the columns even if no matches are found.
    kwic_table = pl.DataFrame(
        kwic_rows,
        schema={
            "Category": pl.String,
            "Keyword": pl.String,
            "n words before kw": pl.String,
            "kw in text": pl.String,
            "n words after kw": pl.String,
        },
    )

    output_path = Path(output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kwic_table.write_csv(output_path, separator=";")

    logger.info(
        "Saved %s KWIC rows to %s",
        kwic_table.height,
        output_path,
    )


