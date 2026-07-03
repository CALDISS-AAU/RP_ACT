# Clean_Combine Pipeline README

## Purpose

The Clean_Combine pipeline prepares raw ACT conversation data for downstream analysis.

The pipeline:

1. Inspects raw conversation datasets.
2. Combines and cleans all raw conversation data.
3. Generates additional working datasets at different aggregation levels.
4. Adds derived variables and contact reasons to each dataset.
5. Exports analysis-ready parquet datasets.
6. Generates logs documenting each processing step.

---

## Input Data

### Conversations

Raw conversation datasets are expected as CSV files located in:

```text
./Data/raw/samtaler
```

### Contact Reasons

Contact reason datasets are expected as CSV files located in:

```text
./Data/raw/emner
```

---

## Pipeline Steps

### 1. Data Inspection

Implemented in:

```text
Functions/data_inspection.py
```

This step:

* Inspects all raw conversation CSV files.
* Logs import warnings.
* Reports row and conversation counts.
* Displays example values from each column.
* Inspects pandas and Python data types.
* Produces a summary across all datasets.

---

### 2. Data Cleaning and Combination

Implemented in:

```text
Functions/raw_data_clean_and_combine.py
```

This step:

* Combines all raw conversation CSV files into a single dataset.
* Removes empty messages.
* Removes non-counselling conversations.
* Keeps only requesters aged 10–17.
* Standardises selected data types.
* Exports the cleaned message-level dataset.

---

### 3. Generate Working Datasets

Implemented in:

```text
Functions/generate_working_datasets.py
```

This step generates additional datasets for downstream analyses:

* **Message dataset** – one row per message.
* **Collapsed message dataset** – consecutive messages from the same sender are merged into a single message.
* **Conversation dataset** – all messages within a conversation are combined into a single row.

Random conversations are logged before and after each transformation to support validation.

---

### 4. Variable Addition

Implemented in:

```text
Functions/add_variables.py
```

Derived variables are added to each dataset, including:

* Age groups.
* Simplified gender.
* Time-of-day categories.
* Seasonal categories.
* Text-based features.
* Contact reasons.

The enriched datasets are exported as parquet files.

---

## Outputs

### Data

The pipeline produces three analysis-ready datasets:

| Dataset                                | Description                                                                                  |
| -------------------------------------- | -------------------------------------------------------------------------------------------- |
| `Data/message_data.parquet`            | Message-level dataset containing one row per message.                                        |
| `Data/collapsed_messages_data.parquet` | Dataset where consecutive messages from the same sender are collapsed into a single message. |
| `Data/conversation_data.parquet`       | Conversation-level dataset containing one row per conversation.                              |

Intermediate datasets are stored in:

```text
Pipelines/Clean_Combine/Data/
```

---

### Logs

Inspection log:

```text
Pipelines/Clean_Combine/Logs/inspection_of_raw_data.log
```

Cleaning log:

```text
Pipelines/Clean_Combine/Logs/clean_and_combine_raw_data.log
```

Cleaning evaluation log:

```text
Pipelines/Clean_Combine/Logs/clean_and_combine_raw_data_evaluation.log
```

Working dataset generation log:

```text
Pipelines/Clean_Combine/Logs/generate_working_datasets.log
```

Variable addition log:

```text
Pipelines/Clean_Combine/Logs/variable_addition.log
```

Combined pipeline log:

```text
Pipelines/Clean_Combine/Logs/full_clean_combine_pipeline.log
```

---

## Running the Pipeline

Run the pipeline from the ACT project root:

```bash
python -m Pipelines.Clean_Combine.clean_combine_main
```
