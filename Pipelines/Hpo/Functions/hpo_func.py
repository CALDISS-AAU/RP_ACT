"""
This script optimizes hyperparameters for a TSDAE model using HPO.
"""

## IMPORTS ##
# External
import sys
import polars as pl
import numpy as np
from datasets import Dataset
from sentence_transformers import models,SentenceTransformer
from sentence_transformers.losses import DenoisingAutoEncoderLoss
from sentence_transformers.datasets import DenoisingAutoEncoderDataset
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer.training_args import BatchSamplers
from torch.utils.data import DataLoader
from polars_splitters import split_into_train_eval
import optuna
# nltk for stuff to work properly
import nltk
nltk.download('punkt_tab')
# Internal
from Shared_Functions.modelling_functions import (
    build_sentence_transformer,
    split_data,
    create_hf_dataset,
    prepare_tsdae_data
)

# STATIC VARIABLES #
def _hpo_loss_init(model):
    return DenoisingAutoEncoderLoss(model, tie_encoder_decoder=True)


def _hpo_search_space(trial):
    return {
        "num_train_epochs": trial.suggest_int(
            "num_train_epochs",
            1,10
            ),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size",
            [16, 32, 64]
            ),
        "warmup_ratio": trial.suggest_float(
            "warmup_ratio",
            0.0,0.3
            ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            1e-6, 5e-5,
            log=True
            ),
        "weight_decay": trial.suggest_float(
            "weight_decay"
            , 0.0 , 0.1
            ),
    }


def _hpo_compute_objective(
    metrics,
    loss,
    input_dir,
    output_dir
     ):
    train_dataset, val_dataset, test_df = prepare_tsdae_data(input_dir, output_dir, log)
    model = build_sentence_transformer()

    # 7. Define the training arguments
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir="checkpoints",
        # Optional training parameters:
        max_steps=10000,  # We might want to limit the number of steps for HPO
        bf16=True,  # Set to True if you have a GPU that supports BF16
        # Optional tracking/debugging parameters:
        eval_strategy="epochs",   # We don't need to evaluate/save during HPO
        save_strategy="no",
        logging_steps=10,
        run_name="tsdae_hpo",  # Will be used in W&B if `wandb` is installed
    )

    # 8. Create the trainer with model_init rather than model
    trainer = SentenceTransformerTrainer(
        model=None,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        evaluator=dev_evaluator,
        model_init=hpo_model_init,
        loss=hpo_loss_init,
    )

    # 9. Perform the HPO
    best_trial = trainer.hyperparameter_search(
        hp_space=hpo_search_space,
        compute_objective=hpo_compute_objective,
        n_trials=20,
        direction="minimize",
        backend="optuna",
    )
    return best_trial
    return metrics["eval_loss"]
