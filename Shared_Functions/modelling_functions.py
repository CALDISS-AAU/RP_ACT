# IMPORTS #
# External
import polars as pl
from datasets import Dataset
from sentence_transformers import SentenceTransformer, models
from polars_splitters import split_into_train_eval

import nltk

nltk.download("punkt_tab")

# Build model baseline


def build_sentence_transformer(
    model_name: str = "BAAI/bge-m3",
    max_seq_length: int = 128,
    pooling_mode: str = "mean",
) -> SentenceTransformer:
    base_model = models.Transformer(model_name, max_seq_length=max_seq_length)

    pooling_model = models.Pooling(
        base_model.get_word_embedding_dimension(),
        pooling_mode=pooling_mode,
    )

    return SentenceTransformer(modules=[base_model, pooling_model])


# Splitter


def split_data(
    df: pl.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    train_df, temp_df = split_into_train_eval(
        df,
        eval_rel_size=val_size + test_size,
        stratify_by=["gender", "age_group"],
        shuffle=False,
        seed=1849,
    )
    val_df, test_df = split_into_train_eval(
        temp_df,
        eval_rel_size=test_size / (val_size + test_size),
        stratify_by=["gender", "age_group"],
        shuffle=False,
        seed=1349,
    )
    return train_df, val_df, test_df


# Create Datasets.Dataset


def create_hf_dataset(df: pl.DataFrame, text_column: str = "msg_tekst") -> Dataset:

    texts = df[text_column].to_list()
    return Dataset.from_dict(
        {
            "source": texts,
            "target": texts,
        }
    )


def prepare_tsdae_data(input_dir):
    dataset = pl.read_parquet(input_dir)
    train_df, val_df, test_df = split_data(dataset)

    train_dataset = create_hf_dataset(train_df)
    val_dataset = create_hf_dataset(val_df)

    return train_dataset, val_dataset, test_df
