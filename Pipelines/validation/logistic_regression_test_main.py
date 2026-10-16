"""Main script for the Logistic_Regression_Test pipeline."""

# External
import pandas as pd
import argparse

# Internal
from Functions.validation_func import _prep_data, _create_report, _LR_gender, _LR_age, _calc_sd_mean


def parse_args():
    parser = argparse.ArgumentParser(description="Run logistic regression")

    parser.add_argument(
    "--calculate-sd-mean",
    action="store_true",
    help="Calculate mean and standard deviation across saved LR reports.",
    )

    parser.add_argument(
        "--results-path",
        type=str,
        default=None,
        help="Path to the LR reports for later reporting"
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
        help="target column in dataset (i.e age_group or gender)"
    )

    parser.add_argument(
    "--seed",
    type=int,
    default=1849,
    help="Random seed used for the train/validation/test split.",
)

    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--class-weight",
        type=str,
        default="balanced",
        choices=["balanced", "none"],
    )

    parser.add_argument(
        "--filename",
        type=str,
        default="confusion_matrix",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    # Calculate mean and standard deviation across saved LR reports
    if args.calculate_sd_mean:
        summary = _calc_sd_mean(results_path=args.results_path)
        print(summary)
        return
    
    # Required for normal LR run
    if args.data_path is None:
        raise ValueError("--data-path is required for LR evaluation.")

    # LR Stuff
    class_weight = None if args.class_weight == "none" else args.class_weight

    df = pd.read_parquet(args.data_path)

    train_df, val_df, test_df = _prep_data(df=df, seed=args.seed)

    # Gender probe
    _, gender_report = _LR_classifier(
        train_df=train_df,
        test_df=val_df,
        random_state=args.seed,
        C=args.C,
        target=args.target,
        class_weight=class_weight,
        filename=f"{args.filename}_{target}_{args.seed}"
    )

    # Age Probe
    _, age_report = _LR_classifier(
        train_df=train_df,
        test_df=val_df,
        random_state=args.seed,
        C=args.C,
        target=args.target,
        class_weight=class_weight,
        filename=f"{args.filename}_{target}_{args.seed}"
    )

    _create_report(
    gender_report=gender_report,
    age_report=age_report,
    filename=f"{args.filename}_{args.seed}.xlsx",
)

if __name__ == "__main__":
    main()
