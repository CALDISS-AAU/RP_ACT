# Conversation_Extraction Pipeline README

## Purpose

The Conversation_Extraction pipeline prepares conversation samples and descriptive summaries from cleaned ACT conversation data.

The pipeline:

1. Investigates the cleaned conversation dataset.
2. Generates descriptive statistics and visualisations.
3. Exports random conversation examples for manual inspection.
4. Extracts stratified conversation samples for qualitative analyses.
5. Generates logs documenting each processing step.

---

## Input Data

The pipeline expects the conversation-level dataset produced by the Clean_Combine pipeline:

```text
Data/conversation_data.parquet
```

---

## Pipeline Steps

### 1. Data Investigation

Implemented in:

```text
Functions/data_investigation.py
```

This step:

* Calculates descriptive wordcount statistics.
* Generates wordcount distribution charts.
* Summarises topic and contact-reason frequencies.
* Investigates topic combinations.
* Calculates wordcount statistics by topic and contact-reason counts.
* Exports a Word report containing tables and figures.

---

### 2. Random Conversation Examples

Implemented in:

```text
Functions/get_random_conversations.py
```

This step:

* Randomly samples conversations with non-empty text.
* Exports each conversation as an individual text file.
* Includes conversation metadata (conversation code, gender, and age group).

These examples are intended for manual inspection and validation of the cleaned dataset.

---

### 3. Stratified Conversation Extraction

Implemented in:

```text
Functions/extract_stratified_conversations.py
```

This step:

* Assigns topics based on conversation contact reasons.
* Stratifies conversations by gender and age group.
* Filters conversations according to configurable wordcount limits.
* Selects conversations to maximise topic coverage while keeping average wordcount close to predefined targets.
* Exports each selected conversation as a Word document.
* Produces a summary of the exported sample.

---

## Outputs

### Data Investigation Report

```text
Output/Conversation_Extraction/data_investiogation_stats.docx
```

Contains descriptive statistics, tables, and charts summarising the conversation dataset.

---

### Random Conversation Examples

```text
Output/Conversation_Extraction/Conversation_Examples/
```

Contains randomly sampled conversations exported as individual text files.

---

### Stratified Conversation Sample

```text
Output/Conversation_Extraction/Stratified_Conversations/
```

Contains:

* Stratified conversation samples organised by gender and age group.
* One Word document per exported conversation.
* An `export_summary.txt` file summarising the final sample.

---

## Logs

Data investigation log:

```text
Pipelines/Conversation_Extraction/Logs/data_investigation.log
```

Random conversation export log:

```text
Pipelines/Conversation_Extraction/Logs/conversation_examples.log
```

Stratified conversation extraction log:

```text
Pipelines/Conversation_Extraction/Logs/stratified_conversation_extraction.log
```

Combined pipeline log:

```text
Pipelines/Conversation_Extraction/Logs/full_conversation_extraction_pipeline.log
```

---

## Running the Pipeline

Run the pipeline from the ACT project root:

```bash
python -m Pipelines.Conversation_Extraction.conversation_extraction_main
```
