"""
This script trains a TSDAE model on RP_ACT data on the message level.
"""

# IMPORTS #
# External
from dataclasses import dataclass
import polars as pl
from sentence_transformers.losses import DenoisingAutoEncoderLoss
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import (
    SentenceTransformerTrainingArguments)

# Internal
from Shared_Functions.modelling_functions import (
    build_sentence_transformer,
    split_data,
    create_hf_dataset,
    load_tsdae_training_data
)

# Classes


@dataclass
class TSDEATrainingConfig:
    """
    Config class for specifying parameters for TSDAE model training
    """

    output_dir: str
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    noise_seed: int = 1998
    del_ratio: float = 0.6

    


# STATIC VARIABLES #


def _build_training_args(
    output_dir,
    epochs,
    batch_size,
    learning_rate,
    warmup_ratio,
    max_grad_norm,
    weight_decay,
):
    return SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        max_grad_norm=max_grad_norm,
        weight_decay=weight_decay,
        bf16=True,
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=500,
        save_total_limit=5,
        logging_strategy="steps",
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=8,
        dataloader_drop_last=True,
    )


def _train_tsdae(
    model,
    train_dataset,
    val_dataset,
    config: TSDEATrainingConfig,
):
    loss = DenoisingAutoEncoderLoss(
        model,
        tie_encoder_decoder=True,
    )

    args = _build_training_args(
        config.output_dir,
        config.epochs,
        config.batch_size,
        config.learning_rate,
        config.warmup_ratio,
        config.max_grad_norm,
        config.weight_decay,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,
    )

    train_result = trainer.train()

    eval_metrics = trainer.evaluate(
        eval_dataset=val_dataset,
    )

    print(f"Training metrics: {train_result.metrics}")
    print(f"Validation metrics: {eval_metrics}")
    print(f"Validation loss: {eval_metrics['eval_loss']:.6f}")

    model.save_pretrained(config.output_dir)

    return model


def _train_model(
    input_dir,
    config: TSDEATrainingConfig,
):
    model = build_sentence_transformer()

    train_dataset, val_dataset, _ = load_tsdae_training_data(
        dataset_path=input_dir,
        del_ratio=config.del_ratio,
        seed=config.noise_seed
    )

    print("TRAIN 1:", train_dataset[0])
    print("TRAIN 2:", train_dataset[0])

    print("VAL 1:", val_dataset[0])
    print("VAL 2:", val_dataset[0])

    model = _train_tsdae(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
    )

    return model
