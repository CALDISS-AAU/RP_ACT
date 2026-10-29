"""Main script for the Logistic_Regression_Test pipeline."""

# External
import pandas as pd
import argparse

# Internal
from Functions.validation_func import (
    _load_split_data,
    _knn_classifier,
    _grind_search,
    _create_report
)

def parse_args():
    parser = argparse.ArgumentParser(description="Run k Nearest Neighbor")

    parser.add_argument(
    "--grid-search-only",
    action="store_true",
    help="Run kNN hyperparameter grid search without final evaluation.",
)
    
    parser.add_argument(
        "--split-dir",
        type=str,
        required=True,
        help="Directory containing train/validation/test Parquet files."
    )

    parser.add_argument(
        "--seed",
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

    # parser.add_argument(
    #     "--target",
    #     type=str,
    #     required=True,
    #     help="target column in dataset (i.e age_group or gender)"
    # )

    parser.add_argument(
        "-k",
        type=int,
        default=5,
        help="Number of nearest neighbors"
    )

    return parser.parse_args()


def main():

    args = parse_args()

    train_df, val_df, test_df = _load_split_data(
        args.split_dir
    )

    if args.grid_search_only:

        gender_grid_search, best_knn_gender = _grind_search(
        train_df=train_df,
        val_df=val_df,
        target="gender",
        filename=f"{args.filename}_seed_{args.seed}"
        )

        age_grid_search, best_knn_age = _grind_search(
            train_df=train_df,
            val_df=val_df,
            target="age_group",
            filename=f"{args.filename}_seed_{args.seed}"
        )
    # Gender test
    _, gender_report = _knn_classifier(
        train_df=train_df,
        val_df=val_df,
        target="gender",
        k=args.k,
        filename=f"{args.filename}_gender",
        output_path=args.output_path
    )

    _, age_report = _knn_classifier(
        train_df=train_df,
        val_df=val_df,
        target='age_group',
        k=args.k,
        filename=f"{args.filename}_age",
        output_path=args.output_path
    )

    _create_report(
    gender_report=gender_report,
    age_report=age_report,
    filename=f"{args.filename}.xlsx",
    )

if __name__ == "__main__":
    main()
