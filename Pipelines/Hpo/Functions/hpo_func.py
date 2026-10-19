"""
This script optimizes hyperparameters for a TSDAE model using Optuna.
"""

# Standard library
import gc
import logging
from pathlib import Path

# External
import optuna
import torch
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from sentence_transformers import SentenceTransformerTrainingArguments
from sentence_transformers.losses import DenoisingAutoEncoderLoss
from sentence_transformers.trainer import SentenceTransformerTrainer

# Internal
from Shared_Functions.modelling_functions import (
    build_sentence_transformer,
    # prepare_tsdae_data,
    load_tsdae_training_data
)
from Pipelines.Data_Prep.Functions.data_func import (
    _prepare_tsdae_data
)


# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

log = logging.getLogger(__name__)


def _hpo_model_init(max_seq_length: int):
    """
    Create a fresh model for every Optuna trial.
    """
    return build_sentence_transformer(
        model_name="BAAI/bge-m3",
        max_seq_length=max_seq_length,
        pooling_mode="mean",
    )


def _hpo_loss_init(model):
    """
    Define a fresh TSDAE loss for every trial.
    """
    loss = DenoisingAutoEncoderLoss(
        model,
        tie_encoder_decoder=True,
    )

    loss.to(model.device)

    return loss


def _hpo_search_space(trial):
    """
    Define the Optuna hyperparameter search space.
    All SentenceTransformerTrainingArguments are valid for this seach space.
    """
    return {
        "num_train_epochs": trial.suggest_int(
            "num_train_epochs",
            3,
            6,
        ),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size",
            [4, 8, 16],
        ),
        "warmup_ratio": trial.suggest_float(
            "warmup_ratio",
            0.0,
            0.15,
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            1e-7,
            5e-4,
            log=True,
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay",
            0.0,
            0.1,
        ),
        "max_grad_norm": trial.suggest_categorical(
            "max_grad_norm",
            [0.5, 1.0, 2.0, 2.5, 3.0],
        ),
    }


def _hpo_compute_objective(metrics):
    """
    Use validation TSDAE loss as the HPO objective.
    """
    return metrics["eval_loss"]


def _run_tsdae_hpo(
    input_dir,
    output_dir,
    log_dir,
    n_trials=4,
    max_seq_length=128,
    study_name="Study_name"
):
    """
    Run the TSDAE Optuna study.

    Parameters
    ----------
    input_dir:
        Path to the input HF dataset.

    output_dir:
        Directory for per-trial Trainer output.

    log_dir:
        Directory for the Optuna journal and text logs.

    n_trials:
        Number of trials requested by this worker.
    """
    input_path = Path(input_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    log_path = Path(log_dir).expanduser().resolve()

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    journal_path = log_path / "tsdae_hpo_journal.log"

    log.info("Input dataset: %s", input_path)
    log.info("Trial output directory: %s", output_path)
    log.info("HPO log directory: %s", log_path)
    log.info("Optuna journal: %s", journal_path)

    train_dataset, val_dataset, _ = load_tsdae_training_data(input_path, del_ratio=0.6, seed=1013)

    storage = JournalStorage(JournalFileBackend(str(journal_path)))

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=storage,
        load_if_exists=True,
    )

    log.info(
        "Loaded Optuna study '%s' with %d existing trials",
        study.study_name,
        len(study.trials),
    )

    def objective(trial):
        params = _hpo_search_space(trial)

        trial_output_path = output_path / f"trial_{trial.number}"

        trial_output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        log.info(
            "Starting Optuna trial %d with parameters: %s",
            trial.number,
            params,
        )

        model = None
        loss = None
        trainer = None

        try:
            model = _hpo_model_init(max_seq_length=max_seq_length)
            loss = _hpo_loss_init(model)

            training_args = SentenceTransformerTrainingArguments(
                output_dir=str(trial_output_path),
                num_train_epochs=params["num_train_epochs"],
                per_device_train_batch_size=params["per_device_train_batch_size"],
                learning_rate=params["learning_rate"],
                warmup_ratio=params["warmup_ratio"],
                weight_decay=params["weight_decay"],
                max_grad_norm=params["max_grad_norm"],
                bf16=True,
                eval_strategy="epoch",
                save_strategy="no",
                logging_steps=50,
                dataloader_num_workers=0,
                dataloader_drop_last=True,
                run_name=f"tsdae_hpo_trial_{trial.number}",
                report_to="none",
            )

            trainer = SentenceTransformerTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                loss=loss,
            )

            trainer.train()

            metrics = trainer.evaluate()

            objective_value = _hpo_compute_objective(metrics)

            log.info(
                "Finished Optuna trial %d with eval_loss=%.6f",
                trial.number,
                objective_value,
            )

            return objective_value

        except torch.cuda.OutOfMemoryError as error:
            log.exception(
                "Trial %d failed with CUDA out-of-memory",
                trial.number,
            )

            raise optuna.TrialPruned("CUDA out-of-memory") from error

        finally:
            if trainer is not None:
                del trainer

            if loss is not None:
                del loss

            if model is not None:
                del model

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    log.info(
        "Starting TSDAE HPO with %d trials for this worker",
        n_trials,
    )

    log.info(
        "Using max_seq_length %d",
        max_seq_length
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=1,
        gc_after_trial=True,
    )

    log.info(
        "Completed trials in study: %d",
        len(study.trials),
    )

    log.info(
        "Best value: %s",
        study.best_value,
    )

    log.info(
        "Best hyperparameters: %s",
        study.best_params,
    )

    return study.best_trial
