"""
This script optimizes hyperparameters for a TSDAE model using HPO.
"""

# IMPORTS ##
# External
import numpy as np
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import DenoisingAutoEncoderLoss
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
import optuna

# Internal
from Shared_Functions.modelling_functions import (
    build_sentence_transformer,
    prepare_tsdae_data
)

# STATIC VARIABLES #

def _hpo_model_init():
    return build_sentence_transformer(
        model_name="BAAI/bge-m3",
        max_seq_length=128,
        pooling_mode="mean",
    )

def _hpo_loss_init(model):
    loss = DenoisingAutoEncoderLoss(model, tie_encoder_decoder=True)
    loss.to(model.device)
    return loss

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
     ):
    """
     choice of hpo compute objective. Several choices are available.
     Since we have unlabelled data for TSDAE - eval_loss is chosen.
    """

    return metrics["eval_loss"]


# def _run_tsdae_hpo(
#     input_dir,
#     output_dir,
#     log
#      ):
#     train_dataset, val_dataset, test_df = prepare_tsdae_data(
#         input_dir
#         )

#     # Define the training arguments
#     args = SentenceTransformerTrainingArguments(
#         output_dir=output_dir,
#         # max_steps=1000,  # We might want to limit the number of steps for HPO
#         bf16=True,

#         eval_strategy="epoch",
#         save_strategy="no",
#         logging_steps=10,
#         run_name="tsdae_hpo",  # Will be used in W&B if `wandb` is installed
#     )

#     # Create the trainer with model_init rather than model
#     trainer = SentenceTransformerTrainer(
#         model=None,
#         args=args,
#         train_dataset=train_dataset,
#         eval_dataset=val_dataset,
#         model_init=_hpo_model_init,
#         loss=_hpo_loss_init
#     )
    

#     # Perform the HPO
#     best_trial = trainer.hyperparameter_search(
#         hp_space=_hpo_search_space,
#         compute_objective=_hpo_compute_objective,
#         n_trials=20,
#         direction="minimize",
#         backend="optuna",
#     )
#     log.info("Running TSDAE HPO")
#     log.info("Best hyperparameters: %s", best_trial.hyperparameters)
#     return best_trial


def _run_tsdae_hpo(input_dir, output_dir, log):

    train_dataset, val_dataset, test_df = prepare_tsdae_data(input_dir)

    def objective(trial):
        model = build_sentence_transformer()

        loss = DenoisingAutoEncoderLoss(
            model,
            tie_encoder_decoder=True,
        )

        args = SentenceTransformerTrainingArguments(
        output_dir=f"{output_dir}/trial_{trial.number}",
        num_train_epochs=trial.suggest_int("num_train_epochs", 1, 10),
        per_device_train_batch_size=trial.suggest_categorical(
            "per_device_train_batch_size", [16, 32, 64]
        ),
        learning_rate=trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True),
        warmup_ratio=trial.suggest_float("warmup_ratio", 0.0, 0.3),
        weight_decay=trial.suggest_float("weight_decay", 0.0, 0.1),
        bf16=True,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        dataloader_num_workers=8,
        dataloader_drop_last=True,
        run_name=f"tsdae_hpo_trial_{trial.number}",
    )

        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            loss=loss,
        )

        trainer.train()
        metrics = trainer.evaluate()
        return metrics["eval_loss"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    log.info("Running TSDAE HPO")
    log.info("Best value: %s", study.best_value)
    log.info("Best hyperparameters: %s", study.best_params)
    return study.best_trial