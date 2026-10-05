"""Main script for the Information Retrieval Evaluation pipeline thingy."""

# External
import argparse
from datasets import load_dataset, load_from_disk

# Internal
from Functions.ire_func import (
    _load_model,
    _create_corpus,
    _InformationRetrievalEvaluator_builder
)

# Internals
from Shared_Functions.modelling_functions import (
    load_tsdae_training_data
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Retrieval Information evaluation")

    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to the IRE results for later reporting"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to SentenceTransformer model",
    )

    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        )

    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to Datasets.dataset"
    )

    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name for IRE evaluation"
    )

    return parser.parse_args()


def main():

    args = parse_args()

    model = _load_model(args.model_path, args.local_files_only)

    corpus, queries, relevant_docs = _create_corpus(args.dataset_path)

    resuls = _InformationRetrievalEvaluator_builder(
        corpus=corpus,
        queries=queries,
        relevant_docs=relevant_docs,
        model=model,
        output_path=args.output_path,
        name=args.name
    )
    

if __name__ == "__main__":
    main()
