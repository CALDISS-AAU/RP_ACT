import logging
import random
from pathlib import Path

from datasets import load_dataset, load_from_disk
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import (
    InformationRetrievalEvaluator,
)

from Shared_Functions.modelling_functions import (
    split_data,
    load_tsdae_training_data,
)

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)


def _load_model(model_path: str, local_files_only: bool = True):
    model = SentenceTransformer(model_path, local_files_only=local_files_only)

    return model


def create_corpus(
    dataset_path: str,
):

    _, val_dataset, _ = load_tsdae_training_data(dataset_path=dataset_path)

    # Create unique ID
    val_dataset = val_dataset.add_column(
    "retrieval_id",
    [f"doc_{i}" for i in range(len(val_dataset))])

    # Convert the datasets to dictionaries
    corpus = dict(
    zip(dataset["retrieval_id"], dataset["text"])
)

    queries = {
        f"q_{rid}": noisy
        for rid, noisy in zip(
            dataset["retrieval_id"],
            dataset["noisy"]
        )
    }

    relevant_docs = {
        f"q_{rid}": {rid}
        for rid in dataset["retrieval_id"]
    }

    relevant_docs = {rid: {rid} for rid in dataset["retrieval_id"]}

    return corpus, queries, relevant_docs

