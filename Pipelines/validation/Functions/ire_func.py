import logging
import random
from pathlib import Path

from datasets import load_dataset, load_from_disk
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import (
    InformationRetrievalEvaluator,
)

from Shared_Functions.modelling_functions import load_tsdae_training_data

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)


def _load_model(model_path: str, local_files_only: bool = True):
    model = SentenceTransformer(model_path, local_files_only=local_files_only)

    return model


def _create_corpus(dataset_path: str):
    """
    loads dataset using load_tsdae_training_data. Performs noise transformation on data and splits text column into (text, noisy).
    default text column is "msg_tekst"

    Creates retrieval IDS for evaluation set
    splits text, noisy into corpus/queries
    maps
    """
    _, val_dataset, _ = load_tsdae_training_data(
        dataset_path=dataset_path
    )

    # Create stable retrieval IDs for this evaluation set
    val_dataset = val_dataset.add_column(
        "retrieval_id",
        [f"doc_{i}" for i in range(len(val_dataset))]
    )

    # Clean documents
    corpus = dict(
        zip(
            val_dataset["retrieval_id"],
            val_dataset["text"],
        )
    )

    # Noisy versions become queries
    queries = {
        f"q_{rid}": noisy
        for rid, noisy in zip(
            val_dataset["retrieval_id"],
            val_dataset["noisy"],
        )
    }

    # Each noisy query is relevant to its original clean document
    relevant_docs = {
        f"q_{rid}": {rid}
        for rid in val_dataset["retrieval_id"]
    }

    return corpus, queries, relevant_docs

def _InformationRetrievalEvaluator_builder(
    model,
    queries,
    corpus,
    relevant_docs,
    name: str,
    output_path: str
):
    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
        batch_size=32,
        show_progress_bar=True,
        write_csv=True
    )
    results = ir_evaluator(model, output_path=output_path)

    print(ir_evaluator.primary_metric)

    print(results[ir_evaluator.primary_metric])

    return results

    