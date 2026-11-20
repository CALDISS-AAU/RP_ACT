"""Extract Danish phrases and export token annotations and phrase counts."""

import csv
import json
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path
import logging
from time import perf_counter

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
    """Build a lowercase phrase from the original token text."""
    return " ".join(token.text.casefold() for token in tokens)


def _get_phrase_tokens(head: Token) -> list[Token]:
    """Collect a head and its directly attached preceding modifiers."""
    context = [
        child
        for child in head.children
        if child.i < head.i
        and (child.pos_, child.dep_) in CONTEXT_RULES
    ]
    return sorted([*context, head], key=lambda token: token.i)


def _analyse_texts(texts, conversation_codes, nlp, logger: logging.Logger):
    """Build CSV rows and collect (primary word, phrase) occurrences."""
    token_rows = []
    occurrences_by_conversation = {}
    occurrences_by_text = {}

    total_texts = len(texts)
    skipped_texts = 0
    started_at = perf_counter()
    logger.info("Text analysis started: %s texts to process.", total_texts)

    for text_index, text in enumerate(texts):
        text_number = text_index + 1
        logger.info(
            "Processing text %s/%s (%.1f%% reached; Text_index=%s).",
            text_number,
            total_texts,
            100 * text_number / total_texts,
            text_index,
        )

        if text is None or not text.strip():
            skipped_texts += 1
            logger.info("Skipped text %s/%s: blank or null.", text_number, total_texts)
            continue

        text_started_at = perf_counter()
        try:
            doc = nlp(text)
        except Exception:
            logger.error(
                "spaCy analysis failed at text %s/%s (Text_index=%s).",
                text_number,
                total_texts,
                text_index,
            )
            raise
        conversation_code = str(conversation_codes[text_index])
        occurrences = occurrences_by_conversation.setdefault(
            conversation_code, []
        )
        text_occurrences = []
        occurrences_by_text[text_index] = text_occurrences

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
            occurrence = (primary_word, normalised_phrase)
            occurrences.append(occurrence)
            text_occurrences.append(occurrence)

        # print(f"Relevante udtryk: {list(phrases_by_head.values())}")

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

        logger.info(
            "Completed text %s/%s (%.1f%%): %s tokens, %s phrase occurrences; %.2f seconds.",
            text_number,
            total_texts,
            100 * text_number / total_texts,
            len(doc),
            len(text_occurrences),
            perf_counter() - text_started_at,
        )

    logger.info(
        "Text analysis finished: %s analysed, %s skipped, %s token rows; %.2f seconds.",
        len(occurrences_by_text),
        skipped_texts,
        len(token_rows),
        perf_counter() - started_at,
    )

    return (
        token_rows,
        occurrences_by_conversation,
        occurrences_by_text,
    )


def _folder_label(value) -> str:
    """Create a safe folder label from a gender or age-group value."""
    if value is None:
        return "__missing__"

    original = str(value)
    label = re.sub(r"[^\w.-]+", "_", original).strip("._")

    # Add a suffix when changing a label, so distinct values stay distinct.
    if not label or label != original or original == "__missing__":
        suffix = hashlib.sha256(
            original.encode("utf-8")
        ).hexdigest()[:10]

        label = f"{label or 'value'}_{suffix}"

    return label


def _iter_strata(df):
    """Yield output subfolders and the original indices of their texts."""
    by_gender = defaultdict(list)
    by_age_group = defaultdict(list)
    by_combination = defaultdict(list)

    metadata = df.select("gender", "age_group")

    for text_index, (gender, age_group) in enumerate(metadata.iter_rows()):
        by_gender[gender].append(text_index)
        by_age_group[age_group].append(text_index)
        by_combination[(gender, age_group)].append(text_index)

    yield Path("Total"), range(df.height)

    for gender, indices in by_gender.items():
        yield Path(f"gender_{_folder_label(gender)}"), indices

    for age_group, indices in by_age_group.items():
        yield Path(f"age_group_{_folder_label(age_group)}"), indices

    for (gender, age_group), indices in by_combination.items():
        folder = (
            Path("gender_age_group")
            / f"gender_{_folder_label(gender)}"
            / f"age_group_{_folder_label(age_group)}"
        )

        yield folder, indices


def _collect_conversation_occurrences(
    text_indices,
    conversation_codes,
    occurrences_by_text,
):
    """Combine selected texts under their conversation codes."""
    occurrences_by_conversation = {}

    for text_index in text_indices:
        # Blank texts were skipped during analysis.
        if text_index not in occurrences_by_text:
            continue

        conversation_code = str(conversation_codes[text_index])

        occurrences = occurrences_by_conversation.setdefault(
            conversation_code, []
        )

        occurrences.extend(occurrences_by_text[text_index])

    return occurrences_by_conversation


def _reduce_to_phrases_on_conversation_level(occurrences_by_conversation):
    """Count normalized phrases under each conversation and primary word."""
    phrases_by_conversation = {}

    for conversation_code, occurrences in occurrences_by_conversation.items():
        counts_by_word = {}

        for primary_word, phrase in occurrences:
            counts = counts_by_word.setdefault(primary_word, Counter())
            counts[phrase] += 1

        phrases_by_conversation[conversation_code] = {
            word: {
                "count": sum(counts.values()),
                "phrases": dict(counts.most_common()),
            }
            for word, counts in counts_by_word.items()
        }

    return phrases_by_conversation


def _section_conversation_summary(
    conversation_summary: dict,
    word_categories: dict,
) -> dict:
    """Group summary entries by category, preserving counts and word order."""
    category_names = (
        "arenas",
        "maybe_arenas",
        "relations",
        "maybe_relations",
        "neither",
    )

    # Convert lists to sets for efficient membership checks.
    category_words = {
        category: {
            word.casefold()
            for word in word_categories[category]
        }
        for category in category_names
    }

    sectioned_summary = {
        category: {}
        for category in category_names
    }

    for primary_word, entry in conversation_summary.items():
        matching_categories = [
            category
            for category in category_names
            if primary_word.casefold() in category_words[category]
        ]

        if not matching_categories:
            # New words may not exist in the original mapping.
            sectioned_summary.setdefault("unmapped", {})[primary_word] = entry
            continue

        for category in matching_categories:
            sectioned_summary[category][primary_word] = entry

    return sectioned_summary
    

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
        primary_word: {
            "count": sum(counts.values()),
            "phrases": dict(counts.most_common()),
        }
        for primary_word, counts in sorted_words
    }


def _summarise_phrases_by_conversation(occurrences_by_conversation):
    """Count conversations per head word and collect unique phrases."""
    summary = {}

    for occurrences in occurrences_by_conversation.values():
        words_in_conversation = set()

        for primary_word, phrase in occurrences:
            entry = summary.setdefault(
                primary_word,
                {"conversation_count": 0, "phrases": set()},
            )
            entry["phrases"].add(phrase)
            words_in_conversation.add(primary_word)

        for primary_word in words_in_conversation:
            summary[primary_word]["conversation_count"] += 1

    # Convert sets to JSON-compatible lists and sort by conversation count.
    return {
        primary_word: {
            "conversation_count": entry["conversation_count"],
            "phrases": sorted(entry["phrases"]),
        }
        for primary_word, entry in sorted(
            summary.items(),
            key=lambda item: item[1]["conversation_count"],
            reverse=True,
        )
    }


def _save_stratum(
    output_dir: Path,
    csv_filename: str,
    token_rows,
    occurrences_by_conversation: dict,
    word_categories: dict,
    logger: logging.Logger,
) -> None:
    """Build and save all outputs for one subset of texts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("[%s] Building phrase counts per conversation.", output_dir)

    phrases_by_conversation = _reduce_to_phrases_on_conversation_level(
        occurrences_by_conversation
    )
    logger.info("[%s] Building global phrase counts.", output_dir)
    summary = _summarise_phrases_across_all_conversations(
        occurrences_by_conversation
    )
    logger.info("[%s] Counting conversations per primary word.", output_dir)
    flat_conversation_summary = _summarise_phrases_by_conversation(
        occurrences_by_conversation
    )
    logger.info("[%s] Grouping words by category.", output_dir)
    conversation_summary = _section_conversation_summary(
        flat_conversation_summary,
        word_categories,
    )
    logger.info("[%s] Building conversation counts without phrases.", output_dir)
    conversation_counts_only = {
        category: {
            primary_word: {
                "conversation_count": entry["conversation_count"],
            }
            for primary_word, entry in entries.items()
        }
        for category, entries in conversation_summary.items()
    }

    csv_file = output_dir / csv_filename

    logger.info("Writing CSV: %s", csv_file)
    with csv_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(CSV_COLUMNS)
        writer.writerows(token_rows)
    logger.info("CSV saved: %s", csv_file)

    json_outputs = {
        "relations_and_arenas_pr_conversation.json": (
            phrases_by_conversation
        ),
        "summary_count.json": summary,
        "summary_conversation_count.json": conversation_summary,
        "summary_conversation_count_without_phrases.json": (
            conversation_counts_only
        ),
    }

    for file_number, (filename, contents) in enumerate(json_outputs.items(), start=1):
        json_file = output_dir / filename
        logger.info("Writing JSON %s/%s: %s", file_number, len(json_outputs), json_file)
        with json_file.open("w", encoding="utf-8") as file:
            json.dump(contents, file, ensure_ascii=False, indent=2)
        logger.info("JSON saved: %s", json_file)


def derive_from_spacy_search(
    input_path: str,
    output_path: str,
    word_categories: dict,
    logger: logging.Logger,
) -> None:
    started_at = perf_counter()
    logger.info("Starting spaCy search.")
    logger.info("Step 1/6: Reading input parquet: %s", Path(input_path).resolve())
    df = pl.read_parquet(input_path)
    # df = df[:10]

    logger.info("Step 2/6: Normalising age-group labels.")
    df = df.with_columns(
        pl.col("age_group")
        .str.strip_chars()
        .str.replace_all(r"\s+", "_")
    )

    logger.info(
        "Input contains %s texts across %s conversation codes.",
        df.height,
        df["conversation_code"].n_unique(),
    )
    logger.info(
        "Metadata: %s rows with null gender; %s rows with null age group.",
        df["gender"].null_count(),
        df["age_group"].null_count(),
    )

    df_texts = df["msg_tekst"]
    conversation_codes = df["conversation_code"].to_list()

    # Build the subsets first, also checking the metadata columns exist.
    logger.info("Step 3/6: Preparing output groups.")
    strata = list(_iter_strata(df))
    logger.info("Prepared %s output groups, including Total.", len(strata))

    logger.info("Step 4/6: Loading spaCy model: da_core_news_trf.")
    model_started_at = perf_counter()
    nlp = spacy.load("da_core_news_trf")
    logger.info("spaCy model loaded in %.2f seconds.", perf_counter() - model_started_at)

    # Analyse every text once.
    logger.info("Step 5/6: Analysing texts.")
    (
        token_rows,
        total_occurrences,
        occurrences_by_text,
    ) = _analyse_texts(
        df_texts,
        conversation_codes,
        nlp,
        logger=logger,
    )

    # Index CSV rows by their original Text_index.
    logger.info("Indexing %s token rows by Text_index.", len(token_rows))
    token_rows_by_text = defaultdict(list)

    for row in token_rows:
        token_rows_by_text[row[0]].append(row)

    logger.info("Token-row indexing completed.")

    output_file = Path(output_path)
    output_root = output_file.parent

    logger.info("Step 6/6: Building and saving outputs to: %s", output_root.resolve())
    for group_number, (folder, text_indices) in enumerate(strata, start=1):
        group_started_at = perf_counter()
        logger.info(
            "Starting output group %s/%s: %s (%s input texts).",
            group_number,
            len(strata),
            folder,
            len(text_indices),
        )
        if folder == Path("Total"):
            selected_occurrences = total_occurrences
            selected_token_rows = token_rows
        else:
            selected_occurrences = _collect_conversation_occurrences(
                text_indices,
                conversation_codes,
                occurrences_by_text,
            )

            # Generate rows as needed instead of copying the whole CSV.
            selected_token_rows = (
                row
                for text_index in text_indices
                for row in token_rows_by_text.get(text_index, ())
            )

        stratum_output_dir = output_root / folder

        logger.info(
            "Saving group %s/%s: %s conversations with analysed text to %s.",
            group_number,
            len(strata),
            len(selected_occurrences),
            stratum_output_dir,
        )

        _save_stratum(
            output_dir=stratum_output_dir,
            csv_filename=output_file.name,
            token_rows=selected_token_rows,
            occurrences_by_conversation=selected_occurrences,
            word_categories=word_categories,
            logger=logger,
        )
        logger.info(
            "Completed output group %s/%s (%.1f%%): %s; %.2f seconds.",
            group_number,
            len(strata),
            100 * group_number / len(strata),
            folder,
            perf_counter() - group_started_at,
        )

    logger.info(
        "spaCy search completed successfully: %s texts analysed, %s groups saved; %.2f seconds total.",
        len(occurrences_by_text),
        len(strata),
        perf_counter() - started_at,
    )
