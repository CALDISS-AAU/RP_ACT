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

