from pathlib import Path
import pandas as pd
import polars as pl
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt


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
    
    return model, report_df

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