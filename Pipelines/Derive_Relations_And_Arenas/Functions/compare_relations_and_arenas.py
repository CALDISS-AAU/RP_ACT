"""Compare conversation-level arena and relation counts across output groups.

Use generate_comparison_workbooks() after derive_from_spacy_search().

Each workbook starts with Summary percentages and Summary ranks. Rows follow
Total order, extending until every selected group's rank-1-to-N words is covered.
Pass top_n and summary_groups from the pipeline main script.
Individual group worksheets are preserved.

Each workbook has one worksheet per group: Total, ages, genders, and
age/gender combinations. Each sheet lists primary words in descending order
of conversation count, with competition ranks, counts, and percentages.

Install the Excel dependency once in the project root:
    uv add openpyxl

Only the confirmed 'arenas' and 'relations' categories are included.
'maybe_arenas' and 'maybe_relations' remain excluded.

Percentages use the number of conversations with at least one analysed
(nonblank) text in the same group, read from the sibling
relations_and_arenas_pr_conversation.json file. Summary counts alone cannot
supply this denominator because one conversation may mention several words.
Missing denominator files leave percentages blank and mark the total as
unavailable, with a logger warning.

Counts and percentages are numeric Excel cells; Excel handles locale formatting.
Percentage formulas recalculate when the workbook opens in Excel.
"""

import re
import json
import logging
from pathlib import Path
from time import perf_counter


SUMMARY_FILENAME = "summary_conversation_count_without_phrases.json"
CONVERSATIONS_FILENAME = "relations_and_arenas_pr_conversation.json"
AGE_ORDER = {"pre_teen": 0, "early_teen": 1, "teen": 2}



def _group_sort_key(folder: Path):
    """Order groups: Total, ages, genders, then gender/age combinations."""
    parts = folder.parts
    if parts == ("Total",):
        return (0, "", 0, "")
    if len(parts) == 1 and parts[0].startswith("age_group_"):
        age = parts[0].removeprefix("age_group_")
        return (1, "", AGE_ORDER.get(age, 99), age)
    if len(parts) == 1 and parts[0].startswith("gender_"):
        return (2, parts[0], 0, "")
    if (
        len(parts) == 3
        and parts[0] == "gender_age_group"
        and parts[1].startswith("gender_")
        and parts[2].startswith("age_group_")
    ):
        age = parts[2].removeprefix("age_group_")
        return (3, parts[1], AGE_ORDER.get(age, 99), age)
    raise ValueError(f"Unrecognised output-group folder: {folder}")


def _read_json_object(path: Path) -> dict:
    """Read a pipeline JSON file and validate its outer structure."""
    with path.open("r", encoding="utf-8-sig") as file:
        contents = json.load(file)
    if not isinstance(contents, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return contents


def _extract_counts(summary: dict, category: str, path: Path) -> dict[str, int]:
    """Read conversation counts without treating malformed values as zero."""
    entries = summary.get(category)
    if not isinstance(entries, dict):
        raise ValueError(f"Missing or invalid category {category!r} in {path}")

    counts = {}
    for word, entry in entries.items():
        count = entry.get("conversation_count") if isinstance(entry, dict) else None
        if type(count) is not int or count < 0:
            raise ValueError(f"Invalid conversation count for {word!r} in {path}")
        counts[word] = count
    return counts


def _ranks(counts: dict[str, int]) -> dict[str, int]:
    """Use competition ranks: tied counts receive ranks such as 1, 1, 3."""
    result = {}
    previous_count = None
    rank = 0
    for position, (word, count) in enumerate(
        sorted(counts.items(), key=lambda item: (-item[1], item[0])), start=1
    ):
        if count == 0:
            continue
        if count != previous_count:
            rank = position
            previous_count = count
        result[word] = rank
    return result


def _sheet_title(label: str, used: set[str]) -> str:
    """Create readable, unique Excel sheet names (maximum 31 characters)."""
    if label == "Total":
        title = "Total"
    else:
        parts = label.split("/")
        if parts[0] == "gender_age_group":
            gender = parts[1].removeprefix("gender_")
            age = parts[2].removeprefix("age_group_")
            title = f"{age} {gender}"
        else:
            title = parts[0].removeprefix("age_group_").removeprefix("gender_")
        title = title.replace("__missing__", "Missing").replace("_", " ").strip()
        title = title[:1].upper() + title[1:]

    title = re.sub(r"[\\/*?:\[\]]", "-", title).strip("'") or "Group"
    base = title[:31]
    candidate = base
    number = 2
    while candidate.casefold() in used:
        suffix = f" ({number})"
        candidate = base[:31 - len(suffix)] + suffix
        number += 1
    used.add(candidate.casefold())
    return candidate


def _summary_data(
    groups: list[dict],
    category: str,
    top_n: int,
    summary_groups: list[tuple[str, str]],
) -> tuple[list[str], list[dict | None], list[list], list[list]]:
    """Extend Total order until every selected group's rank-1-to-N word is included."""
    by_label = {group["label"]: group for group in groups}
    total = by_label["Total"]["counts"][category]
    total_words = sorted(
        (word for word, count in total.items() if count > 0),
        key=lambda word: (-total[word], word),
    )
    selected = [by_label.get(label) for label, _ in summary_groups]
    required_words = set(total_words[:top_n])
    for group in selected:
        if group is not None:
            required_words.update(
                word for word, rank in _ranks(group["counts"][category]).items()
                if 1 <= rank <= top_n
            )

    # Keep every intervening Total row, not just the union of top-N words.
    positions = {word: index for index, word in enumerate(total_words)}
    missing_from_total = required_words.difference(positions)
    if missing_from_total:
        raise ValueError(
            "Group top-N words are absent from Total's positive counts: "
            f"{sorted(missing_from_total)!r}. Check for outputs from different runs."
        )
    last_position = max((positions[word] for word in required_words), default=-1)
    words = total_words[:last_position + 1]
    percentages = [[word] for word in words]
    ranks = [[word] for word in words]
    for group in selected:
        counts = group["counts"][category] if group is not None else {}
        denominator = group["total_conversations"] if group is not None else None
        group_ranks = _ranks(counts)
        for index, word in enumerate(words):
            percentages[index].append(
                counts.get(word, 0) / denominator
                if denominator is not None and denominator > 0 else None
            )
            ranks[index].append(group_ranks.get(word))
    return words, selected, percentages, ranks


def _write_summary_sheet(
    workbook,
    groups: list[dict],
    category: str,
    top_n: int,
    summary_groups: list[tuple[str, str]],
    group_sheets: dict[str, str],
    logger: logging.Logger,
) -> None:
    """Create separate percentage and rank summaries with identical extended rows."""
    from copy import copy
    from openpyxl.styles import Alignment, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.table import Table, TableStyleInfo

    words, selected, percentages, ranks = _summary_data(
        groups, category, top_n, summary_groups,
    )
    last_column = get_column_letter(1 + len(summary_groups))
    headers = ["Primary word", *(caption for _, caption in summary_groups)]
    green_fill = PatternFill("solid", fgColor="E2EFDA")
    yellow_fill = PatternFill("solid", fgColor="FFF2CC")
    red_fill = PatternFill("solid", fgColor="FCE4D6")
    top_n_boundary = min(top_n, len(words))

    # Link finished results from group sheets, instead of repeating calculations.
    source_rows = {}
    for group in selected:
        if group is None:
            continue
        counts = group["counts"][category]
        sorted_words = sorted(counts, key=lambda word: (-counts[word], word))
        source_rows[group["label"]] = {
            word: index for index, word in enumerate(sorted_words, start=7)
        }

    sections = (
        ("Summary percentages", "Conversation percentages", percentages, "D", "0.0%", "percentages"),
        ("Summary ranks", "Word ranks", ranks, "A", "0", "ranks"),
    )
    for sheet_index, (sheet_title, title, data, source_column, number_format, suffix) in enumerate(sections):
        sheet = workbook.create_sheet(sheet_title, sheet_index)
        sheet.sheet_view.showGridLines = False
        sheet.freeze_panes = "B9"
        sheet.merge_cells(f"A1:{last_column}1")
        sheet["A1"] = f"{category.capitalize()} | {title} | Top {top_n}"
        sheet["A1"].font = Font(name="Calibri", size=18, bold=True, color="FFFFFF")
        sheet["A1"].fill = PatternFill("solid", fgColor="17365D")
        sheet.row_dimensions[1].height = 34
        notes = {
            7: "Red word: outside rank 1 to N in every selected group; kept only to preserve the extended Total order.",
            2: "Rows follow Total's count order, extending until every selected group's rank-1-to-N words is included. Ties share a rank and may include more than N words.",
            3: "Green word: rank 1 to N in EVERY selected group. Missing groups prevent green marking.",
            4: "Yellow group cell: this word ranks 1 to N in that group (also shown on percentage cells).",
            5: f"Bold line: end of the original first {top_n} Total rows. Remaining rows complete the selected groups' top-N coverage.",
            6: "Percentages use each group's conversations with analysed text. 0% means no mention. Blank percentage: unavailable/undefined; blank rank: no mention/unavailable.",
        }
        for row, text in notes.items():
            sheet.merge_cells(f"A{row}:{last_column}{row}")
            sheet.cell(row, 1, text).alignment = Alignment(wrap_text=True, vertical="center")
            sheet.row_dimensions[row].height = 32
        sheet["A3"].fill = green_fill
        sheet["A4"].fill = yellow_fill
        sheet["A7"].fill = red_fill
        header_row = 8
        for column, header in enumerate(headers, start=1):
            cell = sheet.cell(header_row, column, header)
            cell.font = Font(name="Calibri", bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="17365D")
            cell.alignment = Alignment(wrap_text=True, vertical="center")
        sheet.row_dimensions[header_row].height = 30

        for data_index, values in enumerate(data):
            row_number = header_row + 1 + data_index
            word = values[0]
            word_cell = sheet.cell(row_number, 1, word)
            word_cell.data_type = "s"
            word_cell.alignment = Alignment(wrap_text=True, vertical="top")
            word_cell.font = Font(name="Calibri", size=11)
            word_ranks = ranks[data_index][1:]
            if all(rank is not None and 1 <= rank <= top_n for rank in word_ranks):
                word_cell.fill = green_fill
            elif not any(rank is not None and 1 <= rank <= top_n for rank in word_ranks):
                word_cell.fill = red_fill
            sheet.row_dimensions[row_number].height = max(
                22, 16 * ((len(word) - 1) // 34 + 1),
            )
            for column, group in enumerate(selected, start=2):
                cell = sheet.cell(row_number, column)
                cell.number_format = number_format
                cell.font = Font(name="Calibri", size=11)
                rank = word_ranks[column - 2]
                if rank is not None and 1 <= rank <= top_n:
                    cell.fill = yellow_fill
                value = values[column - 1]
                if value is None or group is None:
                    continue
                source_row = source_rows[group["label"]].get(word)
                if source_row is None:
                    # A word absent from an available group has percentage zero.
                    cell.value = value
                else:
                    sheet_name = group_sheets[group["label"]].replace("'", "''")
                    cell.value = f"='{sheet_name}'!{source_column}{source_row}"

        if words:
            table = Table(
                displayName=f"summary_{category}_{suffix}",
                ref=f"A{header_row}:{last_column}{header_row + len(words)}",
            )
            # A plain body keeps the green and yellow cell fills visible in Excel.
            table.tableStyleInfo = TableStyleInfo(
                name="TableStyleLight1", showRowStripes=False, showColumnStripes=False,
            )
            sheet.add_table(table)
            boundary_row = header_row + top_n_boundary
            for cell in sheet[boundary_row][:len(headers)]:
                border = copy(cell.border)
                border.bottom = Side(style="thick", color="17365D")
                cell.border = border
        else:
            sheet.cell(header_row + 1, 1, "No words with positive counts in Total.")

        sheet.column_dimensions["A"].width = 38
        for column in range(2, len(headers) + 1):
            sheet.column_dimensions[get_column_letter(column)].width = 16
        sheet.sheet_properties.pageSetUpPr.fitToPage = True
        sheet.page_setup.orientation = "landscape"
        sheet.page_setup.fitToWidth = 1
        sheet.page_setup.fitToHeight = 0
        sheet.print_title_rows = "1:8"
        sheet.print_area = f"A1:{last_column}{max(9, 8 + len(words))}"
        logger.info(
            "Added %s for %s: %s rows, original Total boundary=%s.",
            sheet_title, category, len(words), top_n_boundary,
        )
    workbook.active = 0


def _write_workbook(
    groups: list[dict],
    category: str,
    output_path: Path,
    logger: logging.Logger,
    top_n: int,
    summary_groups: list[tuple[str, str]],
) -> None:
    """Write one compact, independently ranked worksheet per output group."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.table import Table, TableStyleInfo

    workbook = Workbook()
    workbook.remove(workbook.active)
    used_titles = {"summary percentages", "summary ranks"}
    group_sheets = {}
    headers = ["Rank", "Primary word", "Conversation count", "Conversation percentage"]

    for group_number, group in enumerate(groups, start=1):
        title = _sheet_title(group["label"], used_titles)
        logger.info(
            "Writing %s sheet %s/%s: %s", category, group_number, len(groups), title,
        )
        sheet = workbook.create_sheet(title)
        group_sheets[group["label"]] = title
        sheet.sheet_view.showGridLines = False
        sheet.freeze_panes = "C7"
        sheet.sheet_properties.pageSetUpPr.fitToPage = True
        sheet.page_setup.orientation = "portrait"
        sheet.page_setup.paperSize = sheet.PAPERSIZE_A4
        sheet.page_setup.fitToWidth = 1
        sheet.page_setup.fitToHeight = 0
        sheet.print_title_rows = "1:6"
        sheet.print_options.horizontalCentered = True

        denominator = group["total_conversations"]
        sheet.merge_cells("A1:D1")
        sheet["A1"] = f"{category.capitalize()} | {title}"
        sheet["A1"].font = Font(name="Calibri", size=18, bold=True, color="FFFFFF")
        sheet["A1"].fill = PatternFill("solid", fgColor="17365D")
        sheet["A1"].alignment = Alignment(vertical="center")
        sheet.row_dimensions[1].height = 34

        sheet["A2"] = "Group"
        sheet.merge_cells("B2:D2")
        sheet["B2"] = group["label"]
        sheet["B2"].alignment = Alignment(wrap_text=True, vertical="center")
        sheet.row_dimensions[2].height = 32
        sheet.merge_cells("A3:B3")
        sheet["A3"] = "Conversations with analysed text"
        sheet["C3"] = denominator
        if denominator is None:
            sheet["C3"] = "Unavailable"
        sheet["C3"].number_format = "#,##0"
        sheet.merge_cells("A4:D4")
        if denominator is None:
            sheet["A4"] = "Conversation-total JSON missing; percentages are unavailable."
            sheet["A4"].font = Font(name="Calibri", color="9C6500")
        elif denominator == 0:
            sheet["A4"] = "No conversations with analysed text; percentages are undefined."
        else:
            sheet["A4"] = "Percentage = conversations mentioning the word / conversations with analysed text."
        sheet["A4"].alignment = Alignment(wrap_text=True, vertical="center")
        sheet.row_dimensions[4].height = 30
        sheet["A3"].font = Font(name="Calibri", bold=True, color="17365D")

        counts = group["counts"][category]
        ranks = _ranks(counts)
        # Alphabetical order resolves count ties.
        words = sorted(counts, key=lambda word: (-counts[word], word))
        for column, header in enumerate(headers, start=1):
            cell = sheet.cell(6, column, header)
            cell.font = Font(name="Calibri", bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="17365D")
            cell.alignment = Alignment(wrap_text=True, vertical="center")
        sheet.row_dimensions[6].height = 32

        for row_number, word in enumerate(words, start=7):
            sheet.cell(row_number, 1, ranks.get(word))
            word_cell = sheet.cell(row_number, 2, word)
            # Force JSON labels to remain text, even if they start with '='.
            word_cell.data_type = "s"
            word_cell.alignment = Alignment(wrap_text=True, vertical="top")
            sheet.cell(row_number, 3, counts[word]).number_format = "#,##0"
            percent_cell = sheet.cell(row_number, 4)
            if denominator is not None and denominator > 0:
                percent_cell.value = f"=C{row_number}/$C$3"
            percent_cell.number_format = "0.0%"
            for cell in sheet[row_number]:
                cell.font = Font(name="Calibri", size=11)
            sheet.row_dimensions[row_number].height = max(
                20, 16 * ((len(word) - 1) // 34 + 1),
            )

        if words:
            table = Table(displayName=f"{category}_{group_number}", ref=f"A6:D{6 + len(words)}")
            table.tableStyleInfo = TableStyleInfo(
                name="TableStyleMedium2", showFirstColumn=False,
                showLastColumn=False, showRowStripes=True, showColumnStripes=False,
            )
            sheet.add_table(table)
        else:
            sheet["B7"] = "No words in this category for this group."
            sheet["B7"].alignment = Alignment(wrap_text=True)
            sheet.row_dimensions[7].height = 32

        for column, width in {"A": 10, "B": 38, "C": 24, "D": 28}.items():
            sheet.column_dimensions[column].width = width
        sheet.print_area = f"A1:D{max(7, 6 + len(words))}"

    _write_summary_sheet(
        workbook, groups, category, top_n, summary_groups, group_sheets, logger,
    )
    try:
        workbook.save(output_path)
    finally:
        workbook.close()


def generate_comparison_workbooks(
    input_dir: str,
    output_dir: str,
    logger: logging.Logger,
    top_n: int,
    summary_groups: list[tuple[str, str]],
) -> tuple[Path, Path]:
    """Create arenas_comparison.xlsx and relations_comparison.xlsx.

    Args:
        input_dir: Parent directory containing Total, age_group_*, gender_*,
            and gender_age_group folders from the spaCy pipeline.
        output_dir: Directory for the two comparison Excel workbooks.
        logger: Configured logger shared with the pipeline main script.
        top_n: Rank threshold for highlighting and top-word coverage on summaries.
            Rows start with Total's first N words and extend to cover every
            selected group's rank-1-to-N words, including tied ranks.
        summary_groups: Ordered (relative folder, column heading) pairs.
            Supplied by the main script. Missing folders produce blank columns
            and a warning. This affects the summary sheets only, not individual sheets.

    Returns:
        Paths to the arenas and relations workbooks, in that order.

    Notes:
        Each sheet lists only words present in that group, sorted by count.
        Percentages are Excel formulas, displayed with percentage formatting.
        Zero-conversation groups have a blank percentage because it is undefined.
        Existing comparison workbooks are overwritten. All discovered supported group
        folders are read, so remove obsolete outputs before running this function.
    """
    if type(top_n) is not int or top_n < 1:
        raise ValueError("top_n must be a positive integer.")
    summary_groups = list(summary_groups)
    if not summary_groups:
        raise ValueError("summary_groups must contain at least one group.")
    for item in summary_groups:
        if (
            not isinstance(item, (tuple, list)) or len(item) != 2
            or not all(isinstance(value, str) and value.strip() for value in item)
        ):
            raise ValueError("Each summary group must be a (folder, column heading) pair.")
    if len({label for label, _ in summary_groups}) != len(summary_groups):
        raise ValueError("Summary group folders must be unique.")
    if len({caption.casefold() for _, caption in summary_groups}) != len(summary_groups):
        raise ValueError("Summary column headings must be unique.")
    if any(caption.casefold() == "primary word" for _, caption in summary_groups):
        raise ValueError("Summary headings must not duplicate 'Primary word'.")

    started_at = perf_counter()
    input_root = Path(input_dir)
    output_root = Path(output_dir)

    logger.info("Comparison step 1/3: Finding summaries in %s", input_root.resolve())
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")
    if not (input_root / "Total" / SUMMARY_FILENAME).is_file():
        raise FileNotFoundError(f"Required Total summary missing in {input_root / 'Total'}")

    summary_paths = []
    for path in input_root.rglob(SUMMARY_FILENAME):
        folder = path.parent.relative_to(input_root)
        try:
            key = _group_sort_key(folder)
        except ValueError:
            logger.warning("Skipping summary outside the expected group structure: %s", path)
            continue
        summary_paths.append((key, path))
    # Total is always the first group.
    summary_paths.sort(key=lambda item: item[0])
    logger.info("Found %s group summaries.", len(summary_paths))

    logger.info("Comparison step 2/3: Reading counts and conversation totals.")
    groups = []
    for group_number, (_, path) in enumerate(summary_paths, start=1):
        folder = path.parent.relative_to(input_root)
        label = folder.as_posix()
        logger.info("Reading group %s/%s: %s", group_number, len(summary_paths), label)
        summary = _read_json_object(path)
        counts = {
            category: _extract_counts(summary, category, path)
            for category in ("arenas", "relations")
        }

        # Each key in this sibling file represents one conversation in this group.
        conversation_path = path.parent / CONVERSATIONS_FILENAME
        denominator = None
        if conversation_path.is_file():
            denominator = len(_read_json_object(conversation_path))
            for category_counts in counts.values():
                if any(count > denominator for count in category_counts.values()):
                    raise ValueError(
                        f"A word count exceeds the conversation total in {path}. "
                        "Check that the JSON files come from the same pipeline run."
                    )
        else:
            logger.warning(
                "No conversation-total file for %s; percentages will be blank and the total unavailable.",
                label,
            )

        groups.append({"label": label, "counts": counts, "total_conversations": denominator})
        logger.info(
            "Loaded %s: %s arena words, %s relation words; conversation total=%s.",
            label, len(counts["arenas"]), len(counts["relations"]), denominator,
        )

    available = {group["label"] for group in groups}
    for label, caption in summary_groups:
        if label not in available:
            logger.warning(
                "Summary group %s (%s) was not found; its cells will be blank.",
                caption, label,
            )

    logger.info("Comparison step 3/3: Writing Excel workbooks.")
    output_root.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for category in ("arenas", "relations"):
        output_path = output_root / f"{category}_comparison.xlsx"
        _write_workbook(groups, category, output_path, logger, top_n, summary_groups)
        output_paths.append(output_path)
        logger.info("Saved %s with two summary sheets and %s group sheets.", output_path, len(groups))

    logger.info("Comparisons completed in %.2f seconds.", perf_counter() - started_at)
    return output_paths[0], output_paths[1]
