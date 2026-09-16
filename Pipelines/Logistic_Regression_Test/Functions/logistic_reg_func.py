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


def _prep_data(
    df: pd.DataFrame,
    ):
    gender_map = {
        "Pige": "female",
        "Ung kvinde": "female",
        "Dreng": "male",
        "Ung mand": "male",
        "Anden kønsidentitet": "other",
        "Ukendt": "unknown",
        "Ønsker ikke at oplyse": "unknown",
        "Ønsker ikke oplyse køn": "unknown",
    }

    df["gender_clean"] = df["requester_gender_name"].map(gender_map)

    df_model = df[df["gender_clean"].isin(["female", "male"])].copy()

    return df_model


def _logistic_regression(
    df: pd.DataFrame,
    embedding_path: str = '/work/RP_act/ACT/Data/embeddings/collapsed_normalised.npy',
    output_path: str = "/work/RP_ACT/Pipelines/Logistic_Regression_Test/Functions/results",
    filename: str = "Confusion_matrix"
):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    embeddings = np.load(embedding_path)

    X = embeddings[df.index.to_numpy()]
    y = df["gender_clean"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=1998
    )
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
    plt.title("Confusion Matrix")
    plt.tight_layout()
    
    plt.savefig(
    output_path / f"{filename}.png",
    dpi=300,
    bbox_inches="tight",
    )

    plt.close()

    return clf, report


def _create_report(
    report,
    filename: str = "clf_report.xlsx",
    output_path: str = "/work/RP_ACT/Pipelines/Logistic_Regression_Test/Functions/results",
):

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    report_df = pd.DataFrame(report).T
    report_df.to_excel(output_path / filename)
