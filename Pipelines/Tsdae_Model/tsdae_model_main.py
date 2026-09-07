"""Main script for the Tsdae_Model pipeline."""
# External
import argparse
import os
from pathlib import Path

# Internal imports
from Functions.train_func import TSDEATrainingConfig, _train_model

REPO_ROOT = Path(os.environ.get("ACT_REPO_ROOT", ".")).resolve()


def main():
    parser = argparse.ArgumentParser(description="Train TSDAE model")
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to input .parquet file"
    )
    parser.add_argument(
        '--output-dir-logs',
        default=str(REPO_ROOT / "Pipelines" / 'TSDAE_Model' / "Logs"),
        help="Directory for storing training logs",
    )
    parser.add_argument(
        '--output-dir-model',
        default=str(REPO_ROOT / 'Pipelines' / 'TSDAE_Model' / 'modelling'),
        help='Directory for storing trained models',
    )
    # Config parse
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    args = parser.parse_args()

    config = TSDEATrainingConfig(
        output_dir=args.output_dir_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay
    )

    _train_model(
        input_dir=args.data_path,
        config=config
    )


if __name__ == "__main__":
    main()
