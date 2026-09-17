"""Main script for the Sentence_Pair pipeline."""

# Internal imports
# External
import pandas as pd
import polars as pl
from datasets import Dataset, DatasetDict
from Functions.sentence_extraction import (
    _embed_sentences,
    _prepare_seed_candidates,
    _retrieve_top_sent,
)
from Shared_Functions.modelling_functions import split_into_train_eval
# TODO: Brug materiale fra MA og PR til query sætninger
# TODO: query sætning må ik matche mere end 1 sætning fra samme samtale.

def main(
    prepare_seeds=False,
    train_test_split=False
):
    # Data
    document_path = (
        "/work/RP_act/ACT/Data/Clean_Combine/collapsed_messages_data.parquet"
    )

    query_path = (
        "/work/MatiasKokholmAppel#3073/RP_ACT/"
        "Pipelines/Sentence_Pair/Data/"
        "seed_sentences_marked.xlsx"
    )

    collapsed = pd.read_parquet(document_path)

    # -----------------------------------------------------------------------------
    # * Seed sentence extraction. Only run this when in need of new seed candidates
    # -----------------------------------------------------------------------------
    if prepare_seeds:
        seed_candidates = _prepare_seed_candidates(
            df=collapsed,
            # contact_reasons=contact_reasons,
            output_dir="/work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/Sentence_Pair/Data/seeds",
        )

        print(seed_candidates["seed_tag"].value_counts())
        print(f"Total seed candidates: {len(seed_candidates)}")
        return

    # ------------------------------------
    # * Normal pipeline
    # ------------------------------------
    (
        corpus_embeddings,
        query_embeddings,
        corpus_df,
        seed_df,
    ) = _embed_sentences(
        document_path=document_path,
        query_path=query_path,
        text_column="msg_tekst",
    )

    results_df, hits = _retrieve_top_sent(
        corpus_df=corpus_df,
        seed_df=seed_df,
        corpus_embeddings=corpus_embeddings,
        query_embeddings=query_embeddings,
        top_k=200,
    )

    print(results_df.shape)
    results_df = results_df[results_df["score"] < 1.0]

    # datetimes sux
    for col in results_df.select_dtypes(include=["datetimetz"]).columns:
        results_df[col] = results_df[col].dt.tz_localize(None)

    results_df.to_excel(
        "/work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/Sentence_Pair/Data/seeds/seed_topk_results.xlsx"
    )

    # --------------------------------------
    # Train / validation / test split
    # Only run once annotated data is ready
    # --------------------------------------
    if train_test_split:
        annotated_df = pl.read_excel(
            "/work/RP_ACT/Pipelines/Sentence_Pair/Data/seeds/annotated/*"
        )

        train_df, val_df, test_df = split_into_train_eval(
            df=annotated_df
        )

        # Saving split for training
        dataset = DatasetDict(
            "train": Dataset.from_polars(train_df)
            "test": Dataset.from_polars(test_df)
            "validation": Dataset.from_polars(val_df)
        )

        dataset.save_to_disk('sentence_pair_dataset')
        return


if __name__ == "__main__":
    main()
