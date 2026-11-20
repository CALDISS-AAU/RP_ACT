"""Example helper functions for the pipeline."""

## IMPORTS ##
import json
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


def _find_keyword_matches(
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


def _extract_keyword_context(
    doc: Doc,
    token_index: int,
) -> str:
    """Extract the full sentence containing the keyword."""
    keyword_token = doc[token_index]
    sentence = keyword_token.sent

    return sentence.text.strip()


def _extract_relevant_words(
    doc: Doc,
    token_index: int,
    relevant_pos: set[str],
    exclude_stopwords: bool,
    only_alphabetic: bool,
) -> list[str]:
    """Extract relevant lemmas, preserving repetitions and word order."""
    sentence = doc[token_index].sent
    relevant_words = []

    for token in sentence:
        # Exclude the particular keyword occurrence.
        if token.i == token_index:
            continue

        # Whitespace and punctuation are never context words.
        if token.is_space or token.is_punct:
            continue

        if token.pos_ not in relevant_pos:
            continue

        if exclude_stopwords and token.is_stop:
            continue

        if only_alphabetic and not token.is_alpha:
            continue

        lemma = (token.lemma_ or token.text).casefold()

        if lemma:
            relevant_words.append(lemma)

    return relevant_words

    
## ________________ ##

## MAIN FUNCTIONALITY ##
def generate_kwic_table(
    input_dir_data: str,
    input_dir_arenas: str,
    input_dir_relations: str,
    output_dir: str,
    top_relevant_arenas: int,
    top_relevant_relations: int,
    relevant_pos: set[str],
    exclude_stopwords: bool,
    only_alphabetic: bool,
    logger: logging.Logger,
) -> None:
    """Generate one KWIC table with separate arena and relation categories."""
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
    skipped_spacy_rows = 0

    for conversation_number, conversation in enumerate(
        df["msg_tekst"], start=1
    ):
        if conversation is None or not conversation.strip():
            continue

        # Process each conversation once, then search both categories.
        try:
            doc = nlp(conversation)
        except ValueError as exc:
            if not str(exc).startswith("Shape mismatch for blis.gemm:"):
                raise

            skipped_spacy_rows += 1
            logger.exception(
                "Skipping row %s/%s in %s after spaCy failure "
                "(characters=%s, spaCy tokens=%s)",
                conversation_number,
                df.height,
                input_dir_data,
                len(conversation),
                len(nlp.make_doc(conversation)),
            )
            continue

        for category, keywords in keywords_by_category.items():
            matches = _find_keyword_matches(doc, keywords)

            for match in matches:
                token_index = match["token_index"]
                keyword = match["keyword"]

                sentence = _extract_keyword_context(
                    doc=doc,
                    token_index=token_index,
                )

                relevant_words = _extract_relevant_words(
                    doc=doc,
                    token_index=token_index,
                    relevant_pos=relevant_pos,
                    exclude_stopwords=exclude_stopwords,
                    only_alphabetic=only_alphabetic,
                )

                kwic_rows.append({
                    "Category": category,
                    "Keyword": keyword,
                    "sentence": sentence,
                    "kw in text": match["word_in_text"],
                    "relevant_words": json.dumps(
                        relevant_words,
                        ensure_ascii=False,
                    ),
                })

        if conversation_number % 100 == 0:
            logger.info(
                "Reached conversation %s/%s; collected %s KWIC rows",
                conversation_number,
                df.height,
                len(kwic_rows),
            )

    logger.info(
        "Skipped %s of %s input rows due to spaCy shape errors",
        skipped_spacy_rows,
        df.height,
    )

    # An explicit schema preserves the columns even if no matches are found.
    kwic_table = pl.DataFrame(
        kwic_rows,
        schema={
            "Category": pl.String,
            "Keyword": pl.String,
            "kw in text": pl.String,
            "sentence": pl.String,
            "relevant_words": pl.String,
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


