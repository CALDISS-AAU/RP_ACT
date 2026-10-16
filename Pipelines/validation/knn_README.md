# kNN Evaluation Pipeline

## Overview
This pipeline evaluates whether Information such as demographic labels can be recovered from pretrained text embeddings using a k-Nearest Nieghbors (kNN) classifier.

The classifier operates directly on precomputed embedding vectors using cosine distance.
Performance is reported using standard classification metrics.

The pipeline is intended for probing embedding representations against targets such as:
- age_group
- requester_gender_name
- other categorical columns available in the dataset

## Workflow
1. Load the source dataset from parquet.
2. Load or attach precomputed embeddings.
3. Split the data into train, test and validation partitions.
4. Extract the embedding column as model features.
5. Train/fit a KNeighborsClassifier from scikit-learn.
6. Predict the selected target variable. 
7. Generate a classification report.
8. Save the report as CSV.

The current classifier configurations uses:

```python
KNeighborsClassifier(
    n_neighbors=5,
    metric="cosine",
    weights="uniform",
)
```
## Input Requirements

The processed data must contain:
- embedding
- <target column>

The target is supplied through:

```bash
--target
```
Example targets:

```bash
age_group
gender
```

## Running the pipeline

Example:
```bash
uv run knn_main.py \
    --data-path /path/to/data.parquet \
    --embedding-path /path/to/embeddings \
    --results-path /path/to/results \
    --target age_group
```

## CLI Arguments

| Argument              | Description                                                    |
| --------------------- | --------                                                       | 
| --data-path           | Path to the srouce Parquet dataset                             |
| --embedding-path      | Path to precomputed embeddings                                 |
| --results-path        | Directory for evaluation outputs                               |
| --target              | Categorical target column to predict                           |


## Evaluation
The classifier generates a scikit-learn classification_report, including:
* precision

* recall

* F1-score

* support

* accuracy

* macro average

* weighted average

Results are saved as:
```
<filename>_<target>_report.csv
```

## Hyperparameter Search
The helper module also provides a validation-based grid search over:
```python
n_neighbors = [3, 5, 7, 11, 15, 21, 31]
weights = ["uniform", "distance"]
```

Candidate models are evaluated using:

```
macro F1
```

with the existing training and validation partitions preserved through PredifinedSplit.

## Purpose

The pipeline is designed as an embedding probe rather than a downstream production classifier.

Its primary use is to evaluate how strongly categorical attributes, such as gender and age groups, are represented in the embedding space and to compare representation behaviour across the projects trained SentenceTransformers models or embedding configurations.

## OUtput
Example output structure:
```bash
Results/
└── kNN_Results/
    ├── model_age_group_report.csv
    └── model_requester_gender_name_report.csv
```

## Dependencies
List of python dependencies are located in the pyproject.toml file for the full project.

Install using uv approach used by the full ACT project.


## Notes
The current current implementation assumes that embeddings have already been calculated and are stored or joined into the dataframe before classifier training.

Cosine distance us used to keep the probe alligned with similarity-based evaluation of sentence embeddings.