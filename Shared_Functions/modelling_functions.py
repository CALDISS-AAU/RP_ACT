# IMPORTS #
# External
from pathlib import Path
import numpy as np
import faiss
import polars as pl
import pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer, models
from polars_splitters import split_into_train_eval

import nltk
nltk.download('punkt_tab')


def calc_embeddings(
    data_path: str = '/work/RP_act/ACT/Data/Clean_Combine/message_data.parquet',
    model_path: str = "/work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/Tsdae_Model/modelling/ACT-TSDAE-messagelvl",
    output_path: str = "/work/RP_act/ACT/Data/embeddings",
    filename: str = "msg_lvl_embeddings",
    normalize_embeddings: bool = True,
):
    output_path = Path(output_path)

    df = pd.read_parquet(data_path)

    model = SentenceTransformer(model_path, local_files_only=True)

    embeddings = model.encode(
        df["msg_tekst"].tolist(),
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=True,
    ).astype('float32')

    # Save as Numpy array
    np.save(output_path / f"{filename}.npy", embeddings)

    # Save as FAISS because why not?
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(
    index,
    str(output_path / f"{filename}.faiss")
)

def calc_mistral_embeddings(
    data_path: str = '/work/RP_act/ACT/Data/Clean_Combine/message_data.parquet',
    model: str = "mistral-embed",
    api_key: str = os.environ["MISTRAL_API_KEY"],
    dimension: int = 1024,
    output_path: str = "/work/RP_act/ACT/Data/embeddings",
    filename: str = "msg_mistral_embed",
):
    # Ensure output_path is a Path object
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)  # Create dir if it doesn't exist

    # Load data
    df = pd.read_parquet(data_path)

    # Initialize Mistral client
    client = Mistral(api_key=api_key)

    # Fetch embeddings in batches (Mistral API has a limit per request)
    texts = df['msg_tekst'].tolist()
    embeddings = []
    batch_size = 50  # Adjust based on API limits
    
    for i in track(range(0, len(texts), batch_size), description="Generating embeddings..."):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=model,
            inputs=batch,
        )
        embeddings.extend([e.embedding for e in response.data])

    embeddings = np.array(embeddings, dtype=np.float32)  # FAISS requires float32

    # Save as NumPy
    np.save(output_path / f"{filename}.npy", embeddings)

    # Save as FAISS
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(output_path / f"{filename}.faiss"))

    return embeddings.shape # Verify

# Build model baseline
def build_sentence_transformer(
    model_name: str = "BAAI/bge-m3",
    max_seq_length: int = 128,
    pooling_mode: str = "mean",
) -> SentenceTransformer:
    base_model = models.Transformer(
        model_name,
        max_seq_length=max_seq_length
    )

    pooling_model = models.Pooling(
        base_model.get_word_embedding_dimension(),
        pooling_mode=pooling_mode,
    )

    return SentenceTransformer(
        modules=[base_model, pooling_model]
    )

# Splitter


def split_data(
    df: pl.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    train_df, temp_df = split_into_train_eval(
        df,
        eval_rel_size=val_size + test_size,
        stratify_by=['gender', 'age_group'],
        shuffle=False,
        seed=1849
    )
    val_df, test_df = split_into_train_eval(
        temp_df,
        eval_rel_size=test_size / (val_size + test_size),
        stratify_by=['gender', 'age_group'],
        shuffle=False,
        seed=1349
    )
    return train_df, val_df, test_df

# Create Datasets.Dataset


def create_hf_dataset(
    df: pl.DataFrame,
    text_column: str = 'msg_tekst'
) -> Dataset:

    texts = df[text_column].to_list()   
    return Dataset.from_dict({
        "source": texts,
        "target": texts,
    })


def prepare_tsdae_data(input_dir):  
    dataset = pl.read_parquet(input_dir)
    train_df, val_df, test_df = split_data(dataset)

    train_dataset = create_hf_dataset(train_df)
    val_dataset = create_hf_dataset(val_df)

    return train_dataset, val_dataset, test_df
