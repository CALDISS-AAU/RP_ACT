# IMPORTS #
# External
import os
from pathlib import Path
import random
import numpy as np
import faiss
import polars as pl
import pandas as pd
from sentence_transformers import SentenceTransformer, models
from datasets import load_from_disk


import nltk
nltk.download('punkt_tab')


def calc_embeddings(
    data_path: str = '/work/RP_act/ACT/Data/Clean_Combine/message_data.parquet',
    model_path: str = "/work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/Tsdae_Model/modelling/ACT-TSDAE-messagelvl",
    output_path: str = "/work/RP_act/ACT/Data/embeddings",
    filename: str = "msg_lvl_embeddings",
    normalize_embeddings: bool = True,
    local_files_only: bool = True
):
    output_path = Path(output_path)

    df = pd.read_parquet(data_path)

    model = SentenceTransformer(model_path, local_files_only=local_files_only)

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
    api_key: str | None = None,
    dimension: int = 1024,
    output_path: str = "/work/RP_act/ACT/Data/embeddings",
    filename: str = "msg_mistral_embed",
):
    # API key
    if api_key is None:
        api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise ValueError(
            "MISTRAL_API_KEY is required when calculating Mistral embeddings."
        )
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
    max_seq_length: int = 1024,
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






# Create Datasets.Dataset

def noise_transform(
    batch,
    del_ratio: float = 0.6,
    text_column: str = 'msg_tekst'
):
    noisy = []

    for text in batch[text_column]:
        words = text.split()

        if len(words) == 0:
            noisy.append(text)
            continue

        keep_prob = 1.0 - del_ratio

        kept_words = [
            word
            for word in words
            if random.random() < keep_prob
        ]

        # Guarantee at least one token survives
        if len(kept_words) == 0:
            kept_words = [random.choice(words)]

        noisy.append(" ".join(kept_words))

    return {
        "noisy": noisy,
        "text": batch[text_column],
    }

def load_tsdae_training_data(
    dataset_path: str = '/work/RP_act/ACT/Data/Tsdae_Train_Data',
    del_ratio: float = 0.6,
    seed: int = 1998,
):
    dataset = load_from_disk(dataset_path)

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    # Dynamic noise during training
    train_dataset.set_transform(
        lambda batch: noise_transform(
            batch,
            del_ratio=del_ratio,
        ),
        columns=["msg_tekst"],
        output_all_columns=False,
    )

    # Fixed validation corruption
    random.seed(seed)

    val_dataset = val_dataset.map(
        lambda batch: noise_transform(
            batch,
            del_ratio=del_ratio,
        ),
        batched=True,
        remove_columns=val_dataset.column_names
    )

    return train_dataset, val_dataset, test_dataset
