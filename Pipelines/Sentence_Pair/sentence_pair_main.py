"""Main script for the Sentence_Pair pipeline."""

# Internal imports
# External
import pandas as pd
from Functions.sentence_extraction import (
    _embed_sentences, _retrieve_top_sent
)


def main():
    # Data
    document_path = (
        "/work/RP_act/ACT/Data/Clean_Combine/"
        "collapsed_messages_data.parquet"
    )

    query_path = (
        "/work/MatiasKokholmAppel#3073/RP_ACT/"
        "Pipelines/Sentence_Pair/Data/"
        "seed_sentences_marked.xlsx"
    )

    # -----------------------------------------------------------------------------
    # * Seed sentence extraction. Only run this when in need of new seed candidates
    # -----------------------------------------------------------------------------

    # seed_candidates =_prepare_seed_candidates(
    #     df=collapsed,
    #     contact_reasons=contact_reasons,
    #     output_dir="/work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/Sentence_Pair/Data/seeds",
    # )

    # print(seed_candidates["seed_tag"].value_counts())
    # print(f"Total seed candidates: {len(seed_candidates)}")

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
        top_k=250,
    )

    print(results_df.shape)
    results_df = results_df[results_df["score"] < 1.0]
    
    # datetimes sux
    for col in results_df.select_dtypes(include=['datetimetz']).columns:
        results_df[col] = results_df[col].dt.tz_localize(None)

    results_df.to_excel('/work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/Sentence_Pair/Data/seeds/seed_topk_results.xlsx')

if __name__ == "__main__":
    main()
