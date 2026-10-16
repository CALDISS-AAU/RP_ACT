"""Main script for the Logistic_Regression_Test pipeline."""

# External
import pandas as pd
import argparse

# Internal
from Functions.validation_func import (
    _knn_classifier,
    _grind_search,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Run k Nearest Neighbor")

    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to input parquet dataset")

    parser.add_argument(
        "--seed-number",
        type=int,
        default=1849,
        help="Seed number for datasplit subdirectory name"
    )

    parser.add_argument(
        "output-path",
        type=str,
        default=None,
        help="Directory for kNN evaluation results"
    )
    
    parser.add_argument(
        "--filename",
        type=str,
        required=True,
        help="Base filename for evaluation output"
    )

    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="target column in dataset (i.e age_group or gender)"
    )

    parser.add_argument(
        "-k",
        type=int,
        default=5,
        help="Number of nearest neighbors"
    )

    return parser.parse_args()


def main():

    args = parse_args()

    train_df = pd.read_parquet(f"{args.data_path} / seed_{seed_number} / train.parquet")

    val_df = pd.read_parquet(f"{args.data_path} / seed_{seed_number} / val.parquet")

    _,report = _knn_classifier(
        train_df=train_df,
        val_df=val_df,
        target=args.target,
        k=args.k,
        filename=args.filename,
        output_path=args.output_path
    )
    
    print(report)

if __name__ == "__main__":
    main()
