# External packages
import os
from pathlib import Path
import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

# Internal packages
from Shared_Functions.modelling_functions import split_data


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
    X = np.stack([
        np.asanyarray(emb, dtype=np.float32)
        for emb in df['embedding']
    ])

    if not np.isfinite(X).all():
        raise ValueError("Embeddings contain NaN or infinity")
    
    return X


def _LR_gender(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    filename: str,
    C: float = 1.0,
    class_weight="balanced",
    output_path: str = "/work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/validation/Results",
):
    """
    Function for training Logistic regression
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    X_train = _get_embeddings(train_df)
    y_train = train_df["gender"].to_numpy()

    X_test = _get_embeddings(test_df)
    y_test = test_df["gender"].to_numpy()

    # Defining LR
    clf = LogisticRegression(C=C, max_iter=1000, class_weight=class_weight)

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
    plt.title("Confusion Matrix - Gender")
    plt.tight_layout()

    plt.savefig(
        output_path / f"{filename}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return clf, report


def _LR_age(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    filename: str,
    C: float = 1.0,
    class_weight="balanced",
    output_path: str = "/work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/validation/Results",
):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    valid_genders = ['female', 'male']

    train_df = train_df[train_df['gender'].isin(valid_genders)].copy()
    test_df = test_df[test_df['gender'].isin(valid_genders)].copy()

    X_train = np.stack(train_df["embedding"].to_numpy())
    y_train = train_df["age_group"].to_numpy()

    X_test = np.stack(test_df["embedding"].to_numpy())
    y_test = test_df["age_group"].to_numpy()

    LR_age = LogisticRegression(
        C=C,
        max_iter=1000,
        class_weight=class_weight,
    )

    LR_age.fit(X_train, y_train)

    preds_age = LR_age.predict(X_test)

    report = classification_report(
        y_test,
        preds_age,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_test,
        preds_age,
        labels=LR_age.classes_,
    )

    cm_df = pd.DataFrame(
        cm,
        index=LR_age.classes_,
        columns=LR_age.classes_,
    )

    print(cm_df)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=LR_age.classes_,
    )

    disp.plot()
    plt.title("Confusion Matrix - Age")
    plt.tight_layout()

    plt.savefig(
        output_path / f"{filename}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return LR_age, report


def _create_report(
    gender_report,
    age_report,
    filename: str = "lr_results.xlsx",
    output_path: str = "/work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/validation/Results",
):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    gender_df = pd.DataFrame(gender_report).T
    age_df = pd.DataFrame(age_report).T

    save_path = output_path / filename

    print(f"Saving Excel report to: {save_path}")

    with pd.ExcelWriter(save_path) as writer:
        gender_df.to_excel(
            writer,
            sheet_name="Gender",
        )

        age_df.to_excel(
            writer,
            sheet_name="Age",
        )

    print(f"Excel exists: {save_path.exists()}")


def _calc_sd_mean(results_path: str):
    results_path = Path(results_path)

    results = []

    for file in results_path.glob("*.xlsx"):
        report = pd.read_excel(file, index_col=0)

        model = "bge-m3" if "bge" in file.stem.lower() else "tsdae"

        results.append(
            {
                "model": model,
                "file": file.stem,
                "macro_f1": report.loc["macro avg", "f1-score"],
                "balanced_accuracy": report.loc["macro avg", "recall"],
            }
        )

    results_df = pd.DataFrame(results)

    summary = results_df.groupby("model")[["macro_f1", "balanced_accuracy"]].agg(
        ["mean", "std"]
    )

    return summary
