# External packages
import os
from pathlib import Path
import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

# Internal packages
from Shared_Functions.modelling_functions import split_data


def _prep_data(
    df: pd.DataFrame,
    embedding_path: str = '/work/RP_act/ACT/Data/embeddings/collapsed_normalised.npy',
    ):
    """
    Function preps data for logistic regression.
    Inputs:
            df: pandas dataframe 
            embedding_path: path to calculated embeddings
    
    Function does train/test/validation split
    Maps and creates a gender column

    returns:
            df_model: prepped train_df
            val_df: Holdout validation set
            test_df: test set
    """
    embeddings = np.load(embedding_path)
    
    # Merging embeddings back to dataframe
    assert len(df) == len(embeddings)
    df["embedding"] = list(embeddings)

    polars_df = pl.from_pandas(df)
    # splitting
    train_df, val_df, test_df = split_data(polars_df)

    train_df = train_df.to_pandas()
    
    train_df = train_df[train_df['is_incoming'] == True]
    
    df_model = train_df[train_df["gender"].isin(["female", "male"])].copy()

    return df_model, val_df, test_df


def _logistic_regression(
    df: pd.DataFrame,
    output_path: str = "/work/RP_ACT/Pipelines/Logistic_Regression_Test/Functions/results",
    filename: str = "Confusion_matrix"
):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    X = df['embedding'].to_numpy()
    y = df["gender_clean"].to_numpy()

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, stratify=y, random_state=1998
    # )
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)

    report = classification_report(y_test, preds, output_dict=True, zero_division=0)

    cm = confusion_matrix(
    y_test,
    preds,
    labels=clf.classes_
    )

    cm_df = pd.DataFrame(
        cm,
        index=clf.classes_,
        columns=clf.classes_
    )
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
    df: pd.DataFrame,
    embedding_path: str = '/work/RP_act/ACT/Data/embeddings/collapsed_normalised.npy',
    output_path: str = "/work/RP_ACT/Pipelines/Logistic_Regression_Test/Functions/results",
    filename: str = "Confusion_matrix_age"
):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    embeddings = np.load(embedding_path)

    X = df['embedding'].to_numpy()
    y = df["age_group"].to_numpy()

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, stratify=y, random_state=71
    # )
    LR_age = LogisticRegression(max_iter=1000, class_weight='balanced')
    LR_age.fit(X_train, y_train)

    preds_age = LR_age.predict(X_test)

    report = classification_report(y_test, preds_age, output_dict=True, zero_division=0)

    cm = confusion_matrix(
    y_test,
    preds_age,
    labels=LR_age.classes_
    )

    cm_df = pd.DataFrame(
        cm,
        index=LR_age.classes_,
        columns=LR_ageclasses_
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
    report,
    filename: str = "clf_report.xlsx",
    output_path: str = "/work/RP_ACT/Pipelines/Logistic_Regression_Test/Functions/results",
):

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    report_df = pd.DataFrame(report).T
    report_df.to_excel(output_path / filename)
