# External packages
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, balanced_accuracy_score


def _prep_data(df: pd.DataFrame):
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

    df_model = df[df["gender_clean"].isin(["female", "male", "other"])].copy()

    return df_model


def _logistic_regression(
    df: pd.DataFrame,
    model_path: str = "/work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/Tsdae_Model/modelling/ACT-TSDAE-messagelvl",
    normalize_embeddings: bool = True,
):
    model = SentenceTransformer(model_path, local_files_only=True)

    X = model.encode(
        df["msg_tekst"].tolist(),
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=True,
    )
    y = df["gender_clean"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=1998
    )
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)

    report = classification_report(y_test, preds, output_dict=True)

    return clf, report


def _create_report(
    report,
    filename: str = "clf_report.xlsx",
    output_path: str = "/work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/Logistic_Regression_Test/Functions/results",
):

    output_path = Path(output_path)

    report_df = pd.DataFrame(report).T
    report_df.to_excel(output_path / filename)
