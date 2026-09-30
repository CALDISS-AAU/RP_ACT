import logging
import random
from pathlib import Path

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
from datasets import load_dataset

from Shared_Functions.modelling_functions import(
    split_data,
    load_tsdae_training_data,

)

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

def _load_model(
    model_path: str,
    local_files_only: bool = True
):
    model = SentenceTransformer(model_path, local_files_only=local_files_only)
    
    return model

def create_corpus(
    dataset_path: str,
    ):
        _, val_dataset, _ = load_tsdae_training_data(dataset_path=dataset_path)

        
