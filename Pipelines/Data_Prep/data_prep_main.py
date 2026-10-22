"""Main script for the Data_Prep pipeline.

To run this script, use the following command from the project root:
    uv run python -m Pipelines.Data_Prep.data_prep_main
"""

## IMPORTS ##
# External
import argparse, json
from pathlib import Path
import polars as pl

# Internal
from Pipelines.Data_Prep.Functions.data_func import (
    _split_data,
    _create_hf_dataset,
    _prepare_tsdae_data
)

## HELPER FUNCTIONS ##
## ________________ ##


## MAIN FUNCTION ##
def main() -> None:
    """Run the full Data_Prep pipeline."""
    parser = argparse.ArgumentParser(description="Prepare the TSDAE input DatasetDict")
    parser.add_argument("--data-path", required=True, help="Path to parquet file")
    parser.add_argument("--base-dir", default="/work/RP_act/ACT/Data")
    parser.add_argument("--embedding-path", type=str, required=True, help="Path to precalculated embeddings")
    parser.add_argument("--name", default="working_data", help="Subfolder for this version of the prepared data")
    parser.add_argument("--split-seed", type=int, default=1849)
    parser.add_argument("--save-split-parquet", action="store_true")
    args = parser.parse_args()

    dataset = _prepare_tsdae_data(
        data_path=args.data_path,
        base_dir=args.base_dir,
        embeddings_path=args.embedding_path,
        name=args.name,
        seed=args.split_seed,
        save_split_parquets=args.save_split_parquet
    )
    print({k: len(v) for k, v in dataset.items()})

## CALL OF MAIN FUNCTION ##
if __name__ == "__main__":
    main()
