"""Main script for the Sentence_Pair pipeline."""

#Internal imports
from Functions.sentence_extraction import contact_reasons, _prepare_seed_candidates, _embed_sentences



# Data
collapsed = pd.read_parquet(
    "/work/RP_act/ACT/Data/Clean_Combine/collapsed_messages_data.parquet"
)

def main():
    _prepare_seed_candidates(df=collapsed, contact_reasons=contact_reasons, output_dir='/work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/Sentence_Pair/Data/seeds')

    # Load model
    model = SentenceTransformer("/work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/Tsdae_Model/modelling/ACT-TSDAE-Collapsed-v2", local_files_only=True)