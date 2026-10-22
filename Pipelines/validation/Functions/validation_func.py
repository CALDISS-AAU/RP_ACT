#External
import logging
import random
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from mlxtend.plotting import plot_decision_regions

from datasets import load_dataset, load_from_disk
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import (
    InformationRetrievalEvaluator,
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

# Internal
from Shared_Functions.modelling_functions import load_tsdae_training_data
from Pipelines.Data_Prep.Functions.data_func import _split_data

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)


def _load_model(model_path: str, local_files_only: bool = True):
    """
    Load a pretrained SentenceTransformer model.
    input: Path (either local or HF-id) to model
    """
    model = SentenceTransformer(model_path, local_files_only=local_files_only)

    return model


def _create_corpus(dataset_path: str):
    """
    loads dataset using load_tsdae_training_data. Performs noise transformation on data and splits text column into (text, noisy).
    default text column is "msg_tekst"

    Creates retrieval IDS for evaluation set
    splits text, noisy into corpus/queries
    maps
    """
    _, val_dataset, _ = load_tsdae_training_data(
        dataset_path=dataset_path,
        preserve_columns=True
    )

    # Create stable retrieval IDs for this evaluation set
    val_dataset = val_dataset.add_column(
        "retrieval_id",
        [f"doc_{i}" for i in range(len(val_dataset))]
    )

    # Clean documents
    corpus = dict(
        zip(
            val_dataset["retrieval_id"],
            val_dataset["text"],
        )
    )

    # Noisy versions become queries
    queries = {
        f"q_{rid}": noisy
        for rid, noisy in zip(
            val_dataset["retrieval_id"],
            val_dataset["noisy"],
        )
    }

    # Each noisy query is relevant to its original clean document
    relevant_docs = {
        f"q_{rid}": {rid}
        for rid in val_dataset["retrieval_id"]
    }

    return val_dataset, corpus, queries, relevant_docs

def _create_group_ir_data(dataset, group_column: str):
    corpus = dict(
        zip(
            dataset["retrieval_id"],
            dataset["text"],
        )
    )

    groups = {}
    
    group_values = sorted(
        value
        for value in set(dataset[group_column])
        if value is not None
    )
    for group_value in group_values:
        group_dataset = dataset.filter(
            lambda x: x[group_column] == group_value
        )

        queries = {
            f"q_{rid}": noisy
            for rid, noisy in zip(
                group_dataset["retrieval_id"],
                group_dataset["noisy"],
            )
        }

        relevant_docs = {
            f"q_{rid}": {rid}
            for rid in group_dataset["retrieval_id"]
        }

        groups[group_value] = {
            "queries": queries,
            "relevant_docs": relevant_docs,
            "n_queries": len(group_dataset),
        }

    return corpus, groups


def _InformationRetrievalEvaluator_builder(
    model,
    queries,
    corpus,
    relevant_docs,
    name: str,
    output_path: str,
    batch_size: int = 32
):
    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
        batch_size=batch_size,
        show_progress_bar=True,
        write_csv=True
    )
    results = ir_evaluator(model, output_path=output_path)

    print(ir_evaluator.primary_metric)

    print(results[ir_evaluator.primary_metric])

    return results


# k Nearest Neighbors functionality
def _knn_classifier(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target: str,
    filename: str,
    output_path: str = (
        "/work/MatiasKokholmAppel#3073/RP_ACT/"
        "Pipelines/validation/Results/kNN_Results"
    ),
    k: int = 5,
): 
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Training
    X_train = np.stack(train_df["embedding"].to_numpy())
    X_val = np.stack(val_df["embedding"].to_numpy())

    # Targets
    y_train = train_df[target]
    y_val = val_df[target]

    model = KNeighborsClassifier(
        n_neighbors=k,
        metric="cosine",
        weights="uniform"
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    report = classification_report(
        y_val,
        predictions,
        zero_division=0,
        output_dict=True
    )

    report_df = pd.DataFrame(report).T

    report_df.to_csv(
        output_path / f"{filename}_{target}_report.csv"
    )

    cm = confusion_matrix(y_test, preds, labels=model.classes_)

    cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
    print(cm_df)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=model.classes_
    )

    disp.plot()
    plt.title(f"Confusion Matrix - {target}")
    plt.tight_layout()

    plt.savefig(
        output_path / f"{filename}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    
    return model, report_df

def knn_plots(
    model,
    report
):
    return monke

def _grind_search(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target: str,
):
    X_train = np.stack(train_df["embedding"].to_numpy())
    X_val = np.stack(val_df["embedding"].to_numpy())

    y_train = train_df[target].to_numpy()
    y_val = val_df[target].to_numpy()

    # Combine train and validation
    X = np.vstack([X_train, X_val])
    y = np.concatenate([y_train, y_val])

    # -1 means training, 0 means validation
    test_fold = np.concatenate([
        np.full(len(X_train), -1),
        np.zeros(len(X_val), dtype=int)
    ])

    cv = PredefinedSplit(test_fold)

    knn = KNeighborsClassifier(
        metric="cosine",
        n_jobs=-1
    )

    param_grid = {
        "n_neighbors": [3, 5, 7, 11, 15, 21, 31],
        "weights": ["uniform", "distance"]
    }

    grid = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=2,
        verbose=1,
        refit=False
    )

    grid.fit(X, y)

    print("Best parameters:", grid.best_params_)
    print("Validation F1:", grid.best_score_)

    return grid

# Logistic Regression functionality

def _load_split_data(split_dir: str | Path):
    """
    Load existing train/validation/test splits
    and prepare them for Logistic Regression.
    """
    split_dir = Path(split_dir)

    train_df = pl.read_parquet(split_dir / "train.parquet")
    val_df = pl.read_parquet(split_dir / "val.parquet")
    test_df = pl.read_parquet(split_dir / "test.parquet")

    required_columns = ['embedding', 'age_group', 'gender']

    for name, df in [
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ]:
        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"Missing columns in {name}: {missing}"
            )

    valid_genders = ["female", "male"]

    train_df = train_df.filter(
        pl.col("gender").is_in(valid_genders)
    )
    val_df = val_df.filter(
        pl.col("gender").is_in(valid_genders)
    )
    test_df = test_df.filter(
        pl.col("gender").is_in(valid_genders)
    )

    return (
        train_df.to_pandas(),
        val_df.to_pandas(),
        test_df.to_pandas(),
    )


def _prep_data(df: pd.DataFrame, seed: int):
    """
    Function preps data for logistic regression.
    Inputs:
            df: pandas dataframe with embedding column

    Function does train/test/validation split
    Maps and creates a gender column

    returns:
            df_model: prepped train_df
            val_df: Holdout validation set
            test_df: test set
    """
    print("df rows:", len(df))

    required_columns = [
        'embedding',
        'age_group',
        'gender'
        ]
    
    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError (f"Missing columns: {missing}")

    polars_df = pl.from_pandas(df)

    # splitting
    train_df, val_df, test_df = split_data(polars_df, seed=seed)

    train_df = train_df.to_pandas()
    val_df = val_df.to_pandas()
    test_df = test_df.to_pandas()

    valid_genders = ["female", "male"]

    train_df = train_df[train_df["gender"].isin(valid_genders)].copy()

    val_df = val_df[val_df["gender"].isin(valid_genders)].copy()

    test_df = test_df[test_df["gender"].isin(valid_genders)].copy()

    return train_df, val_df, test_df

def _get_embeddings(df: pd.DataFrame):
    """
    Pulls embeddings from a dataframe and saves it as X for classification
    input: pandas dataframe
    output: embeddings in a stack numpy array.
    """
    X = np.stack([
        np.asanyarray(emb, dtype=np.float32)
        for emb in df['embedding']
    ])

    if not np.isfinite(X).all():
        raise ValueError("Embeddings contain NaN or infinity")
    
    return X


def _LR_classifier(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    filename: str,
    target: str,
    seed: int,
    C: float = 1.0,
    class_weight="balanced",
    output_path: str = "/work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/validation/Results/LR_results",
):
    """
    Function for training Logistic regression
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    X_train = _get_embeddings(train_df)
    y_train = train_df[target].to_numpy()

    X_test = _get_embeddings(val_df)
    y_test = val_df[target].to_numpy()

    # Defining LR
    clf = LogisticRegression(C=C, max_iter=1000, class_weight=class_weight, random_state=seed)

    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)

    report = classification_report(y_test, preds, output_dict=True, zero_division=0)

    cm = confusion_matrix(y_test, preds, labels=clf.classes_)

    cm_df = pd.DataFrame(cm, index=clf.classes_, columns=clf.classes_)
    print(cm_df)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=clf.classes_,
    )

    disp.plot()
    plt.title(f"Confusion Matrix - {target}")
    plt.tight_layout()

    plt.savefig(
        output_path / f"{filename}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return clf, report