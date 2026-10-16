# Information Retrieval Evaluation Pipeline

## Overview

This pipeline evaluates a SentenceTransformer / TSDAE model using an Information Retrieval Evaluator (IR) task.

The evaluation treats:

- the noisy version of each validation sample as a query
- corresponding clean text as the relevant document
- all clean validation texts as the retrieval corpus

The main goal is to measure how well the model retrieves the original clean text from its noisy counterpart.

The pipeline also supports subgroup evaluation, specifically:

- gender
- age_group

Subgroup queries are evaluated against the same full validation corpus to keep retrieval difficulty comparable across groups.

## Workflow

The evaluation pipeline performs the following steps:

1. Load validation dataset using the shared function load_tsdae_training_dataset.
2. Apply the predifined fixed noise transformation.
3. Add stable retrieval IDs to each validation sample.
4. Constructs:

   - corpus
   - queries
   - relevant-document mappings (relevant_docs)
5. Load target SentenceTransformer model
6. Run InformationRetrievalEvaluator
7. Save evaluation results as CSV.
8. Optionally runs the subgroup evaluation across target subgroups (intended groups are gender and age_groups)

## ire_func.py

Contains the core IR functionality:

```python
_load_model()
_create_corpus()
_create_group_ir_data()
_InformationRetrievalEvaluator_builder()
```

## InformationRetrievalEval_main.py

main script that handles:

- CLI arguments
- model loading
- dataset loading
- overall evaluation
- subgroup evaluation
- relevant output paths

## Dataset requirements

The validation dataset should contain at least:

- text
- noisy

where:

- text is the clean/origninal doc entry
- noisy is the transformed fixed noisy counterpart used as the query

For subgroup evaluation, the dataset should atleast contain:

- gender
- age_group

note:
This transformation is created in another pipeline. Specifically Data_Prep pipeline.

## Retrieval IDs

The pipeline relies on creating independent retrieval IDs instead of relying on source-specific identifiers such as conversation UDs or message numbers.

### Example:

```python
val_dataset = val_dataset.add_column(
    "retrieval_id",
    [f"doc_{i}" for i in range(len(val_dataset))]
)
```

This ensures that every retrieval document has a unique identifier.

Queries use a separate namespace:

```
document: doc_42
query: q_doc_42
```

## corpus construction

The clean validation texts from the corpus is created:

```python
corpus = dict(
    zip(
        val_dataset["retrieval_id"],
        val_dataset["text"],
    )
)
```

Example:

```python
{
    "doc_0": "Some clean document...",
    "doc_1": "Another clean document...",
}
```

## Query Construction

The noisy versions form the queries:

```python
queries = {
    f"q_{rid}": noisy
    for rid, noisy in zip(
        val_dataset["retrieval_id"],
        val_dataset["noisy"],
    )
}
```

Example:

```python
{
    "q_doc_0": "Noisy version...",
    "q_doc_1": "Another noisy version...",
}
```

## Relevant Document Mapping

Eeach noisy query has exactly one known relevant clean document thus creating the structure:

```python
{
    "q_doc_0": {"doc_0"},
    "q_doc_1": {"doc_1"},
}
```

## Model Evaluation

Evaluation is performed using:

```
InformationRetrievalEvaluator
```

Example Confuguration:

```python
ir_evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name=name,
    batch_size=32,
    show_progress_bar=True,
    write_csv=True,
) 
```

Evaluator is called:

```python
results = ir_evaluator(model, output_path=output_path)
```

## Metrics

The evaluator reports several Information Retrieval metrics. Below is an explanation for each type.
Since the project uses normalised embeddings the default primary metric used are the cosine equivalent version of each metric.

### Accuracy / Recall@k

Recall is defined as:
![alt text](image.png)

Recall is the fraction of the documents that are relevant to the query that are successfully retrieved. Since the model is based on the text/noisy only one is correct. The k refers to which rank the correct text was placed. In other words, it measures wheter the relevant clean document appears in the top k retrieved documents.
Because this evaluation contains one relevant document per query:

```
Recall@1
```

caon be interpreted as:

> Percentage of noisy queries where the original clean document was retrieved as rank 1.

Example:

```math
Recall@1 = 0.52
```

means that approximately 52% of noisy queries retrieved their clean document first.

## Precision@k

Measures the proportion of retrieved documents in the top K that are relevant.

Because there is only one relevant document per query, Precision@K naturally decreases as K increases.

## MRR@k

Mean Reciprocal Rank measures the process that produces a list of possible responses to a sample of queries, ordered by probability of correctness. The reciprocal rank of a query response is the multiplicative inverse of the rank of the first correct answer:

1. for first place
2. ½ for sencond place
3. 1⁄3 for third place

it is defined as:
![alt text](image-1.png)

Mean Reciprocal Rank thus measures how early the first relevant document appears.

Higher values indicate that the correct document tends to appear near the top of the ranking.

## NDCG@K

Normalized Discounted Cumulative Gain rewards relevant documents appearing near the top of the retrieval ranking. nDCG is often used to measure effectiveness of seach engine algorithms and related applications where a graded relevance scale of docuemnts used. This is useful to get an idea of how well the correct answer is actually placed, when it is not the first one.

DCG or nDCG has two assumptions:

1. Highly relevant documents are more useful when they appear earlier in a result list (Higher rank).
2. Highly relevant documents are more useful than marginally relevant documents, which are more useful than non-relevant documents.

But, this should still be read in the context of this pipeline where only one answer is correct. Since the test is based on the assumption of how to well a trained SentenceTransformer is able to reconstruct a given docuemnt, based on a query.

## MAP@k

Mean Average Precision summarizes retrieval quality across queries.

With only one relevant document per query, MAP behaves similarly to reciprocal-rank-based metrics.

# Baseline comparison

The same validation set and fixed noisy queries should be used for both the baseline encoder and the TSDAE-trained model.

Example reporting format:


| Model        | Recall@k | MRR@K | nDCG@10 | MAP@100 |
| ------------ | -------- | ----- | ------- | ------- |
| Baseline     | ...      | ...   | ...     | ...     |
| TSDAE-models | ...      | ...   | ...     | ...     |

## Subgroup Evaluation

The Pipeline can additionally evaluate retrieval perofrmance by demographic subgroups.

Currently supported examples include:

```latex
gender
age_group
```

Each subgroup contains only queries belonging to that group, while retrieval is still performed against the **full validation corpus**.

Conceptually this means:

```python
female queries ---\
male queries -------+--> same full document corpus
other queries ------/
```

This avoids changing the difficultiy of retrieval problem between groups.

This also ensures within group reporting for each sub-subgroup, i.e. male, female, preteen, teen etc.

## CLI Arguments


| Argument              | Required | Description                                                    |
| --------------------- | -------- | -------------------------------------------------------------- |
| --model-path          | Yes      | Local path or Hugging Face ID of the SentenceTransformer model |
| --dataset-path        | Yes      | Path to the dataset used for validation                        |
| --output-path         | Yes      | Directory where evaluation CSV files are written               |
| --name                | Yes      | Name assigned to the evaluation run                            |
| --local-files-only    | No       | Restrict model loading to local files                          |
| --no-local-files-only | No       | Allow remote model loading                                     |

## Output

with:

```python
write_csv=True
```

the evaluator automatically writes metric results to supplied output directory.

Example:

```latex
results/
└── ir_evaluation/
    ├── Information-Retrieval_evaluation_message_WoA_1849_eval_results.csv
    ├── Information-Retrieval_evaluation_tsdae_validation_gender_female_results.csv
    ├── Information-Retrieval_evaluation_tsdae_validation_gender_male_results.csv
    └── ...
```

Exact filesnames depend on the evaluator version and supplied evaluation name.

## Reproducibility

For meaningful comparison between models:

* use the same validation split
* use the same fixed noise transformation
* use the same retrieval corpus
* use the same subgroup definitions
* use the same similarity function
* use the same evaluation parameters

The retrieval IDs should also be generated deterministically from the unchanged validation set.

## Interpretation

This evaluation measures whether the embedding model maps a noisy version of a document close to its original clean version.

It should therefore be interpreted primarily as a measure of:

```latex
noise robustness
+
representation consistency
+
retrieval quality
```

It is not necessarily equivalent to a conventional semantic search benchmark, where queries and relevant documents may contain different wording or express related but non-identical information.

## Notes & Limitation

Potential limitations include:

* only one known relevant document per query
* subgroup sample sizes may be highly imbalanced
* small demographic groups may produce unstable estimates
* the evaluation is directly tied to the chosen noise transformation
* strong performance on noisy-to-clean retrieval does not necessarily imply equally strong performance on every downstream semantic retrieval task

Additional robustness tests may therefore be useful.



## Future work

Possible extensions may include:

* evaluate multiple random seeds
* calculate subgroup performance gaps
* calculate confidence intervals
* compare additional demographic or metadata groups (time of writing, seasons etc.)
* test multiple noise severities
* evaluate on an external semantic-search benchmark
* save all subgroup metrics into one consolidated CSV
* automatically generate evaluation plots and reports



## Dependencies

List of python dependencies are located in the pyproject.toml file for the full project.

Install using uv approach used by the full ACT project.

## References

* Sentence Transformers documentation
* `InformationRetrievalEvaluator`
* TSDAE: Transformer-based Sequential Denoising Auto-Encoder for Unsupervised Sentence Embedding Learning
