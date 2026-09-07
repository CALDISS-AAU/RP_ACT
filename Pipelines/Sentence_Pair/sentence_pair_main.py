"""Main script for the Sentence_Pair pipeline."""

# Internal imports
# External
import pandas as pd
from Functions.sentence_extraction import (
    _embed_sentences,
)

# Data
collapsed = pd.read_parquet(
    "/work/RP_act/ACT/Data/Clean_Combine/collapsed_messages_data.parquet"
)


def main():
    # seed_candidates =_prepare_seed_candidates(
    #     df=collapsed,
    #     contact_reasons=contact_reasons,
    #     output_dir="/work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/Sentence_Pair/Data/seeds",
    # )

    # print(seed_candidates["seed_tag"].value_counts())
    # print(f"Total seed candidates: {len(seed_candidates)}")

    # Load model
    model = SentenceTransformer("/work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/Tsdae_Model/modelling/ACT-TSDAE-Collapsed-v2", local_files_only=True)

    corpus_embeddings, query_embeddings  = _embed_sentences()


if __name__ == "__main__":
    main()
