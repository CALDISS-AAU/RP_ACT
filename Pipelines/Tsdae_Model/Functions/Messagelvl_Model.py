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
from sklearn.model_selection import train_test_split

import nltk
nltk.download('punkt_tab')

## STATIC VARIABLES ##
# Directories
INPUT_DIR_DATA = "/work/RP_act/ACT/Pipelines/Clean_Combine/Data/combined_data.parquet"

OUTPUT_DIR_LOG_TRAIN = "/work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/Tsdae_Model/Logs/trainer.log"
OUTPUT_DIR_MODEL = '/work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/Tsdae_Model/ACT_messagelvl_embed'

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

def create_strata_col(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
    (
        pl.col("requester_gender_name")
         + "_" + 
         pl.col("age_group")
         ).alias("strata"),
)

def split_data(
        df: pl.DataFrame,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:

    train_df, temp_df = train_test_split(
        df,
        test_size= (val_size + test_size),
        stratify=df['strata'],
        random_state=1849,
    )
    test_df, val_df = train_test_split(
        temp_df,
        test_size= test_size / (val_size + test_size),
        random_state=1349,
        stratify=df['strata'],
    )
    return train_df, test_df, val_df

def train_tsdae(model, train_df,val_df, output_dir,logging_dir, epochs=5, batch_size=32, lr=2e-5, warmup_ratio=0.1, max_grad_norm=1.0):
    loss = DenoisingAutoEncoderLoss(model, tie_encoder_decoder=True)

    args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR_MODEL,
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
        logging_dir=OUTPUT_DIR_LOG_TRAIN,

        dataloader_num_workers=8

    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_df,
        eval_dataset=val_df,
        loss=loss,
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR_MODEL)
    return model
