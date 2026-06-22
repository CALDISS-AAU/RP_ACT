"""
This script trains a TSDAE model on RP_ACT data on the message level.
"""

## IMPORTS ##
# External
import sys
import polars as pl
import numpy as np
from sentence_transformers import models,SentenceTransformer
from sentence_transformers.losses import DenoisingAutoEncoderLoss
from sentence_transformers.datasets import DenoisingAutoEncoderDataset
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from torch.utils.data import DataLoader
from polars_splitters import split_into_train_eval

import nltk
nltk.download('punkt_tab')

## STATIC VARIABLES ##

def _build_sentence_transformer(
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

def _create_strata_col(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
    (
        pl.col("requester_gender_name")
         + "_" + 
         pl.col("age_group")
         ).alias("strata"),
)

def _split_data(
    df: pl.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    train_df, temp_df = split_into_train_eval(
        df,
        eval_rel_size = val_size + test_size,
        stratify_by= ['requester_gender_name', 'age_group'],
        shuffle=False,
        seed=1849
    )
    val_df, test_df = split_into_train_eval(
        temp_df,
        eval_rel_size = test_size / (val_size + test_size),
        stratify_by = ['requester_gender_name', 'age_group'],
        shuffle=False,
        seed=1349
    )
    return train_df, val_df, test_df


def _create_hf_dataset(
    df: pl.DataFrame,
    text_column: str = 'message_tekst'
) -> Dataset: 
    return Dataset.from_dict({'text': df[text_column].to_list()})

def _train_tsdae(
    model,
    train_dataset,
    val_dataset,
    output_dir,
    logging_dir,
    epochs=5,
    batch_size=32,
    lr=2e-5,
    warmup_ratio=0.1,
    max_grad_norm=1.0):
    loss = DenoisingAutoEncoderLoss(model, tie_encoder_decoder=True)

    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs = epochs,
        per_device_train_batch_size = batch_size,
        learning_rate = lr,
        warmup_ratio=warmup_ratio,
        bf16=True,
        max_grad_norm=max_grad_norm,

        save_strategy="steps",
        save_steps=500,
        save_total_limit=5,
        logging_strategy='steps',
        logging_steps=50,
        logging_dir=logging_dir,

        dataloader_num_workers=8

    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    return model

def train_message_model(input_dir, output_dir, log):
    model = _build_sentence_transformer()
    
    dataset = pl.read_parquet(input_dir)
    train_df, val_df, test_df = _split_data(dataset)
    train_dataset = _create_hf_dataset(train_df)

    val_dataset = _create_hf_dataset(val_df)

    model = _train_tsdae(
        model,
        train_dataset,
        val_dataset,
        output_dir,
        log
        )