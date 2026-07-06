# Clean_Combine Pipeline README

## Purpose

The Clean_Combine pipeline prepares ACT conversation data for downstream analysis.

The pipeline:

1. Inspects raw conversation datasets.
2. Combines all raw datasets into a single dataframe.
3. Applies data cleaning and type standardisation.
4. Adds derived variables and metadata.
5. Exports a final analysis-ready dataset.
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

The inspection step:

* Opens all raw datasets.
* Logs import warnings.
* Reports row and conversation counts.
* Shows example values from each column.
* Inspects pandas and Python data types.
* Produces a summary across all datasets.

### 2. Data Cleaning and Combination

Implemented in:

```text
Functions/raw_data_clean_and_combine.py
```

The cleaning step:

* Combines all raw CSV files.
* Removes empty messages.
* Removes non-counselling conversations.
* Keeps only users aged 10–17.
* Converts selected columns to appropriate datatypes.
* Exports the cleaned dataset as parquet.

### 3. Variable Creation

Implemented in:

```text
Functions/add_variables.py
```

The feature engineering step adds:

* Age groups.
* Time-of-day categories.
* Seasonal categories.
* Text length features.
* Contact reasons.

The enriched dataset is exported as parquet.

---

## Outputs

### Data

Intermediate dataset:

```text
Pipelines/Clean_Combine/Data/cleaned_data.parquet
```

Final dataset:

```text
Data/clean_combined_data.parquet
```

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

Variable creation log:

```text
Pipelines/Clean_Combine/Logs/variable_addition.log
```

Combined pipeline log:

```text
Pipelines/Clean_Combine/Logs/full_clean_combine_pipeline.log
```

---

## Running the Pipeline

Run from the ACT project root:

```bash
python -m Pipelines.Clean_Combine.clean_combine_main
```
