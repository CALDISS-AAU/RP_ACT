import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import util

"""Da pipeline
Seed sentence
    ↓
TSDAE embedding
    ↓
Cosine similarity against corpus embeddings
    ↓
Top-k nearest sentences
    ↓
Candidate sentences for your evaluation dataset
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


def _prepare_seed_candidates(df, contact_reasons, output_dir):
   df_exploded = df.explode("contact_reasons").copy()

   df_reasons = df_exploded[
        df_exploded["contact_reasons"].isin(contact_reasons)
    ].copy()

   for col in df_reasons.select_dtypes(include=["datetimetz"]).columns:
        df_reasons[col] = df_reasons[col].dt.tz_localize(None)

   df_reasons.to_excel(f"{output_dir}/seed_sentences.xlsx", index=False)

   return df_reasons


def _embed_sentences(
    df,
    model,
    text_column,
    batch_size=64,
):
    df = df.copy()

    sentences = df[text_column].fillna("").astype(str).tolist()

    embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return df, embeddings
