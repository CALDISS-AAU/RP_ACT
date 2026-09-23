"""Main script for the Logistic_Regression_Test pipeline."""

# External
import pandas as pd
import argparse

# Internal
from Functions.logistic_reg_func import _prep_data, _create_report, _LR_gender, _LR_age

# Internals
from Shared_Functions.modelling_functions import (
    calc_embeddings,
    # calc_mistral_embeddings,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run logistic regression")

    parser.add_argument(
        "--calculate-embeddings",
        action="store_true",
        help="Recalculate embeddings before running",
    )

    parser.add_argument(
        "--embedding-path",
        type=str,
        required=True,
        help="Path to calculated embeddings used for validation pipeline",
    )

    parser.add_argument("--data-path", type=str, required=True, help="Path to data")

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

    # Embedding stuff
    if args.calculate_embeddings:
        calc_embeddings(
            data_path="/work/RP_act/ACT/Data/Clean_Combine/message_data_without_advisors.parquet",
            filename="msg_lvl_without_advisors_embeddings",
            normalize_embeddings=True,
        )

        calc_embeddings(
            data_path="/work/RP_act/ACT/Data/Clean_Combine/message_data.parquet",
            filename="msg_lvl_embeddings",
            normalize_embeddings=True,
        )

        calc_embeddings(
            model_path="/work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/Tsdae_Model/modelling/ACT-TSDAE-Collapsed-v2",
            data_path="/work/RP_act/ACT/Data/Clean_Combine/collapsed_messages_data_without_advisors.parquet",
            filename="collapsed_without_advisors",
            normalize_embeddings=True,
        )

        calc_embeddings(
            model_path="/work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/Tsdae_Model/modelling/ACT-TSDAE-Collapsed-v2",
            data_path="/work/RP_act/ACT/Data/Clean_Combine/collapsed_messages_data.parquet",
            filename="collapsed_embeddings",
            normalize_embeddings=True,
        )

        calc_embeddings(
            data_path="/work/RP_act/ACT/Data/Clean_Combine/message_data_without_advisors.parquet",
            model_path="BAAI/bge-m3",
            filename="bge_m3_msg_without_advisors_embeddings",
            local_files_only=False,
        )

        # calc_mistral_embeddings()

    # LR Stuff
    class_weight = None if args.class_weight == "none" else args.class_weight

    df = pd.read_parquet(args.data_path)

    train_df, val_df, test_df = _prep_data(df=df, embedding_path=args.embedding_path, seed=args.seed)

    # Gender probe
    _, gender_report = _LR_gender(
        train_df=train_df,
        test_df=test_df,
        C=args.C,
        class_weight=class_weight,
        filename=f"{args.filename}_gender",
    )

    # Age Probe
    _, age_report = _LR_age(
        train_df=train_df,
        test_df=test_df,
        C=args.C,
        class_weight=class_weight,
        filename=f"{args.filename}_age"
    )

    _create_report(
    gender_report=gender_report,
    age_report=age_report,
    filename=f"{args.filename}.xlsx",
)


if __name__ == "__main__":
    main()
