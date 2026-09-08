from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sentence_transformers import (
    SentenceTransformer,
    SimilarityFunction,
    util,
)


"""
Selected well-being-related tags
        ↓
Select ~10 diverse seed sentences per tag
        ↓
Seed sentences
        ↓
Encode with TSDAE model
        ↓
Cosine similarity against
all corpus embeddings
        ↓
Retrieve top-k neighbours

        ↓
Candidate sentence pairs
(seed, retrieved sentence)
        ↓
Stratified selection of candidates
        ↓
Human annotation
according to well-being framework
        ↓
Final evaluation dataset
        ↓
Compare:
Base model vs TSDAE model
"""

"""
All relevant contact-reason tags
        ↓
select diverse seed sentences
        ↓
TSDAE retrieval across corpus
        ↓
candidate sentence pairs
        ↓
human / framework-based labeling
        ↓
split into:
    train
    validation
    test
        ↓
supervised contrastive / pairwise fine-tuning
        ↓
final evaluation on untouched test set
"""


device = "cuda" if torch.cuda.is_available() else "cpu"


def _prepare_seed_candidates(df, output_dir):
    contact_reasons = (
        df['contact_reasons']
        .explode()
        .dropna()
        .unique()
        .tolist()
        )
    
    print(f"Number of contact reasons: {len(contact_reasons)}")

    df_exploded = df.explode("contact_reasons").copy()

    df_reasons = df_exploded[
        df_exploded["contact_reasons"].isin(contact_reasons)
    ].copy()

    df_reasons = df_reasons[df_reasons["is_incoming"]].copy()

    candidate_pools = []

    for tag in contact_reasons:
        tagged_messages = df_reasons[df_reasons["contact_reasons"] == tag]

        sampled = tagged_messages.sample(
            n=min(150, len(tagged_messages)), random_state=55
        ).copy()

        # Makes explicit which tag this sentence was sampled for
        sampled["seed_tag"] = tag

        candidate_pools.append(sampled)

        seed_candidates = pd.concat(candidate_pools, ignore_index=True)

    for col in seed_candidates.select_dtypes(include=["datetimetz"]).columns:
        seed_candidates[col] = seed_candidates[col].dt.tz_localize(None)

    seed_candidates["selected_seed"] = ""

    seed_candidates.to_excel(f"{output_dir}/seed_sentences.xlsx", index=False)

    return seed_candidates


def _embed_sentences(
    document_path,
    query_path,
    text_column='msg_tekst',
    batch_size=64,
    model_name="/work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/Tsdae_Model/modelling/ACT-TSDAE-Collapsed-v2",
):
    model = SentenceTransformer(
        model_name,
        device=device,
        similarity_fn_name=SimilarityFunction.DOT_PRODUCT
    )

    # Corpus
    corpus_df = (
        pd.read_parquet(document_path)
        .dropna(subset=[text_column])
        .copy()
    )

    corpus_df[text_column] = (
        corpus_df[text_column]
        .astype(str)
        .str.strip()
    )

    corpus_df = (
        corpus_df[corpus_df[text_column] != ""]
        .reset_index(drop=True)
    )
    print(corpus_df.columns.tolist())
    print("text_column:", repr(text_column))
    corpus = corpus_df[text_column].tolist()

    # Seeds
    seed_df = pd.read_excel(query_path)

    seed_df = (
        seed_df[seed_df["selected_seed"] == True]
        .copy()
        .reset_index(drop=True)
    )

    queries = seed_df[text_column].tolist()

    # Embeddings
    corpus_embeddings = model.encode_document(
        corpus,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    query_embeddings = model.encode_query(
        queries,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    return (
        corpus_embeddings,
        query_embeddings,
        corpus_df,
        seed_df,
    )


def _retrieve_top_sent(
    corpus_df,
    seed_df,
    corpus_embeddings,
    query_embeddings,
    top_k=250,
):
    hits = util.semantic_search(
        query_embeddings,
        corpus_embeddings,
        top_k=top_k,
    )

    results = []

    for query_id, query_hits in enumerate(hits):
        seed_row = seed_df.iloc[query_id]

        for hit in query_hits:
            corpus_id = hit["corpus_id"]
            corpus_row = corpus_df.iloc[corpus_id]

            results.append(
                {
                    "query_id": query_id,
                    "corpus_id": corpus_id,
                    "score": hit["score"],
                    "seed_text": seed_row["msg_tekst"],
                    "seed_tag": seed_row["seed_tag"],
                    **corpus_row.to_dict(),
                }
            )

    results_df = pd.DataFrame(results)

    return results_df, hits
