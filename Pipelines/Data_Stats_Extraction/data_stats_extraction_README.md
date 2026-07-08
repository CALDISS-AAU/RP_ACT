# Data_Stats_Extraction README

## Overview

The **Data_Stats_Extraction** pipeline generates descriptive statistics for the cleaned ACT datasets. It is intended to provide an overview of the processed data for quality assurance, documentation, and decision-making.

The pipeline analyzes three datasets:

* **Original messages** – Statistics for all individual messages.
* **Collapsed messages** – Statistics after consecutive messages from the same sender have been merged.
* **Conversations** – Statistics at the conversation level.

The resulting statistics are combined into a single JSON report, while detailed logs are generated for each processing step.

## Input

The pipeline expects the cleaned datasets produced by the **Clean_Combine** pipeline:

* `message_data.parquet`
* `collapsed_messages_data.parquet`
* `conversation_data.parquet`

## Output

The pipeline produces:

* `data_stats.json` – A JSON report containing descriptive statistics for all three datasets.
* Individual log files for each processing step.
* A combined pipeline log.

## Statistics Included

Depending on the dataset, the pipeline reports:

* Number of messages and/or conversations.
* Character and word count statistics (average, median, minimum, maximum).
* Conversation duration statistics.
* Distributions of demographic and metadata variables, including:

  * Age group
  * Gender
  * Platform
  * Time of day
  * Season
  * Contact topics
  * Number of contact topics per observation

## Running the Pipeline

Run the pipeline from the ACT project root:

```bash
python -m Pipelines.Data_Stats_Extraction.data_stats_extraction_main
```
