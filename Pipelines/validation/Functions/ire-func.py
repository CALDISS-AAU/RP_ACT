import logging
import random
from pathlib import Path

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
from datasets import load_dataset

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

def _prepare_data_nigga(
    data_path: str
):
    data_path = Path(data_path)

    