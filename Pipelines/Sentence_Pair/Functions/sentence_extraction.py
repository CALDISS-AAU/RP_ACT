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
Seed
  ↓
TSDAE embedding
  ↓
Cosine similarity against corpus
  ↓
Retrieve e.g. top 200
  ↓
Remove:
  • exact duplicates
  • near-duplicates
  • seed itself
  ↓
Sample across similarity/rank ranges
  ↓
Candidate pairs
"""

contact_reasons = [
    "kroppens udseende og udvikling",
    "forældres krav og forventninger",
    "religion/kultur",
    "ensomhed",
    "trivsel i klassen/skolen",
    "flirt/forelskelse",
    "faglige problemer/lektier",
    "stress/præstationsangst",
    "gaming",
    "brug af alkohol",
]

device = "cuda" if torch.cuda.is_available() else "cpu"

def _prepare_seed_candidates(df, contact_reasons, output_dir):
   df_exploded = df.explode("contact_reasons").copy()

   df_reasons = df_exploded[
        df_exploded["contact_reasons"].isin(contact_reasons)
    ].copy()

   df_reasons = df_reasons[
        df_reasons["is_incoming"]
    ].copy()

   candidate_pools = []
   
   for tag in contact_reasons:
        tagged_messages = df_reasons[
            df_reasons["contact_reasons"] == tag
        ]

        sampled = tagged_messages.sample(
            n=min(150, len(tagged_messages)),
            random_state=55
        ).copy()

        # Makes explicit which tag this sentence was sampled for
        sampled["seed_tag"] = tag

        candidate_pools.append(sampled)

        seed_candidates = pd.concat(
            candidate_pools,
            ignore_index=True
        )

   for col in seed_candidates.select_dtypes(include=["datetimetz"]).columns:
        seed_candidates[col] = seed_candidates[col].dt.tz_localize(None)
    
   seed_candidates['selected_seed'] = ''

   seed_candidates.to_excel(f"{output_dir}/seed_sentences.xlsx", index=False)

   return seed_candidates


def _embed_sentences(
    document_path,
    query_path,
    text_column,
    batch_size=64,
    model_name: str = '/work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/Tsdae_Model/modelling/ACT-TSDAE-Collapsed-v2'
):
    model = SentenceTransformer(model_name, device=device, similarity_fn_name=SimilarityFunction.DOT_PRODUCT)
    
    
    
    # Corpus
    corpus = pd.read_parquet(document_path).copy()

    corpus_df = (
    pd.read_parquet(document_path)
    .dropna(subset=[text_column])
    .copy()
    )

    corpus_df[text_column] = (corpus_df[text_column].astype(str).str.strip())

    corpus_df = corpus_df[corpus_df[text_column] != ""].reset_index(drop=True)

    corpus = corpus_df[text_column].tolist()

    # Seeds
    seed_sentences = pd.read_excel(query_path)
    
    queries = seed_sentences[
    seed_sentences["selected_seed"] == True
    ].copy()

    queries = queries["text"].tolist()
    
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
        normalize_embeddings=True
    )


    return corpus_embeddings, query_embeddings, corpus, seed_sentences

def _retrieve_top_sent(
        corpus_embeddings,
        query_embeddings,
        output_dir,
        model_name: str,
        top_k: int = 200
        ):

        hits = util.semantic_search(
                query_embeddings,
                corpus_embeddings,
                top_k=top_k
        )

        corpus_id = hits[0][0]['corpus_id']

        matched_row = corpus_df.iloc(corpus_id)

        
