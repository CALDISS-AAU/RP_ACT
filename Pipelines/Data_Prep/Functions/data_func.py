import os
from pathlib import Path
import numpy as np
import polars as pl
from polars_splitters import split_into_train_eval
from datasets import Dataset, DatasetDict, load_from_disk


# Splitter
def _split_data(
    df: pl.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 1849,
    save_dir: str | Path | None = None,
    dataset_name: str | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Adds precalculated embedddings to DataFrame.
    Split a Polars DataFrame into stratified train, validation, and test sets.

    Optionally save Parquet splits under:
        save_dir / dataset_name / seed_{seed}/

    The data is stratified by gender and age group to preserve their
    distributions across the resulting splits. The seed controls the
    random shuffling and makes the split reproducible.

    Args:
        df: DataFrame to split.
        val_size: Proportion of the data assigned to the validation set.
        test_size: Proportion of the data assigned to the test set.
        seed: Random seed used for reproducible splitting.

    Returns:
        A tuple containing the train, validation, and test DataFrames,
        respectively.
    """

    train_df, temp_df = split_into_train_eval(
        df,
        eval_rel_size=val_size + test_size,
        stratify_by=["gender", "age_group"],
        shuffle=True,
        seed=seed,
    )
    val_df, test_df = split_into_train_eval(
        temp_df,
        eval_rel_size=test_size / (val_size + test_size),
        stratify_by=["gender", "age_group"],
        shuffle=True,
        seed=seed,
    )

    if save_dir is not None:
        if not dataset_name:
            raise ValueError("dataset_name is required when saving splits")

        split_dir = Path(save_dir) / dataset_name / f"seed_{seed}"
        split_dir.mkdir(parents=True, exist_ok=True)

        train_df.write_parquet(split_dir / "train.parquet")
        val_df.write_parquet(split_dir / "val.parquet")
        test_df.write_parquet(split_dir / "test.parquet")

        print(f"Splits saved to: {split_dir}")

    return train_df, val_df, test_df


# Convert to DatasetDict
def _create_hf_dataset(df: pl.DataFrame, text_column: str = "msg_tekst") -> Dataset:

    columns = [text_column, "gender", "age_group"]

    return Dataset.from_dict({col: df[col].to_list() for col in columns})


def _prepare_tsdae_data(
    data_path: str,
    embeddings_path: str,
    base_dir: str | Path,
    name: str,
    seed: int = 1849,
    save_split_parquets: bool = False,
) -> DatasetDict:

    data_path = Path(data_path)

    # Hugging Face DatasetDict destination
    output_dir = Path(base_dir) / "Tsdae_Train_Data" / name

    dataset = pl.read_parquet(data_path)

    embeddings = np.load(embeddings_path)
    if len(dataset) != len(embeddings):
        raise ValueError(
            f"Dataset has {len(dataset)} rows, "
            f"embeddings contain {len(embeddings)} rows"
        )

    # Merging embeddings to dataframe

    dataset = dataset.with_columns(pl.Series("embedding", embeddings.tolist()))

    # Add embeddings, split data, and optionally save Parquets
    train_df, val_df, test_df = _split_data(
        df=dataset,
        seed=seed,
        save_dir=(Path(base_dir) / "splits" if save_split_parquets else None),
        dataset_name=data_path.stem,
    )

    # Create Hugging Face DatasetDict
    hf_dataset = DatasetDict(
        {
            "train": _create_hf_dataset(train_df),
            "validation": _create_hf_dataset(val_df),
            "test": _create_hf_dataset(test_df),
        }
    )

    hf_dataset.save_to_disk(str(output_dir))

    print(f"Hugging Face dataset saved to: {output_dir}")

    return hf_dataset
