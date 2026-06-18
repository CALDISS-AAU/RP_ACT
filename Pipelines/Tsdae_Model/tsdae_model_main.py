"""Main script for the Tsdae_Model pipeline."""
# Internal imports 
from Messagelvel_Model import create_strata_col, split_data, build_sentence_transformer, train_tsdae, INPUT_DIR_DATA
# Helper functions
def _setup_inspection_logger(output_dir_log: str) -> logging.Logger:
    """Set up and return a file logger for data inspection.

    Creates the log directory if it does not exist and configures
    a logger that writes to output_dir_log.

    Args:
        output_dir_log: Full directory where the log file is stored.

    Returns:
        Configured logger instance.
    """
    log_file = Path(output_dir_log)

    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("inspection_of_raw_data")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, mode="w")
        formatter = logging.Formatter("%(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

logger = _setup_inspection_logger(output_dir_log)

def main():
    model = build_sentence_transformer(model_name)
    dataset = read_parquet(INPUT_DIR_DATA)
    dataset = create_strata_col(dataset)
    train, test = split_data(dataset)
    model = train_tsdae(model, dataset)


if __name__ == "__main__":
    main()

