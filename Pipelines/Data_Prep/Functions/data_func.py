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
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Split a Polars DataFrame into stratified train, validation, and test sets.

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
        stratify_by=['gender', 'age_group'],
        shuffle=True,
        seed=seed
    )
    val_df, test_df = split_into_train_eval(
        temp_df,
        eval_rel_size=test_size / (val_size + test_size),
        stratify_by=['gender', 'age_group'],
        shuffle=True,
        seed=seed
    )
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        train_df.write_parquet(save_dir / "train.parquet")
        val_df.write_parquet(save_dir / "val.parquet")
        test_df.write_parquet(save_dir / "test.parquet")

    return train_df, val_df, test_df

# Convert to DatasetDict
def _create_hf_dataset(
    df: pl.DataFrame,
    text_column: str = "msg_tekst"
    ) -> Dataset:
    
    return Dataset.from_dict({text_column: df[text_column].to_list()})


def _prepare_tsdae_data(
    data_path: str,
    base_dir: str | Path,
    embeddings_path: str,
    name: str = "working_data",
    seed: int = 1849,
    save_split_parquets: bool = False,
)-> DatasetDict:

    data_path = Path(data_path)
    output_dir = Path(base_dir) / name
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = pl.read_parquet(data_path)
    
    embeddings = np.load(embedding_path)
    print("df rows:", len(dataset))
    print("embedding rows:", len(embeddings))

    # Merging embeddings back to dataframe
    assert len(df) == len(embeddings), (
        "Number of dataframe rows does not match number of embeddings"
    )

    dataset = dataset.copy()
    dataset["embedding"] = list(embeddings)

    train_df, val_df, test_df = _split_data(
        dataset,
        seed=seed,
        save_dir=Path(output_dir) / "splits" if save_split_parquets else None)

    dataset = DatasetDict({
        "train": _create_hf_dataset(train_df),
        "validation": _create_hf_dataset(val_df),
        "test": _create_hf_dataset(test_df)
    })
    dataset.save_to_disk(output_dir)

    return dataset