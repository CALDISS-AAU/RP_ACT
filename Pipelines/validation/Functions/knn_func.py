from pathlib import Path
import pandas as pd
import polars as pl
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt


def knn_classifier(
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
        age_predictions,
        zero_division=0,
        output_dict=True
    )

    report_df = pd.DataFrame(report).T

    report_df.to_csv(
        output_path / f"{filename}_{target}_report.csv"
    )
    
    return model, report_df
