"""Main script for the Logistic_Regression_Test pipeline."""

# External
import pandas as pd
import argparse

# Internal
from Functions.knn_func import _knn_classifier

def parse_args():
    parser = argparse.ArgumentParser(description="Run k Nearest Neighbor")

    parser.add_argument(
        "--results-path",
        type=str,
        default=None,
        help="Path to the LR reports for later reporting"
    )

    parser.add_argument(
        "--embedding-path",
        type=str,
        default=None,
        help="Path to calculated embeddings used for validation pipeline",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to data"
    )

    
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="target column in dataset (i.e age_group or requester_gender_name)"
    )

    return parser.parse_args()


def main():

    args = parse_args()

    df = pd.read_parquet(args.data_path)

    train_df, val_df, test_df = _prep_data(df=df, embedding_path=args.embedding_path, seed=args.seed)

    # Gender probe
    _, gender_report = _LR_gender(
        train_df=train_df,
        test_df=test_df,
        C=args.C,
        class_weight=class_weight,
        filename=f"{args.filename}_gender_{args.seed}",
    )

    # Age Probe
    _, age_report = _LR_age(
        train_df=train_df,
        test_df=test_df,
        C=args.C,
        class_weight=class_weight,
        filename=f"{args.filename}_age_{args.seed}"
    )

    _create_report(
    gender_report=gender_report,
    age_report=age_report,
    filename=f"{args.filename}_{args.seed}.xlsx",
)

if __name__ == "__main__":
    main()
