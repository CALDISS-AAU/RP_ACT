## IMPORTS ##
# External
import logging
from pathlib import Path
import pandas as pd
import warnings
# Internal
## _______ ##

## VARIABLES TO BE CHANGED ##


## _______________________ ##

## HELPER FUNCTIONS ##
def setup_inspection_logger(output_dir_log: str) -> logging.Logger:
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

def extract_info(dataframe: pd.DataFrame, logger: logging.Logger) -> dict:
    """Extract relevant information from dataframe 
       and writes it to the log.
    """
    logger.info("The columns within the dataframe and their relevant dtypes are:")

    column_type_info = {}

    for col, dtype in dataframe.dtypes.items():
        logger.info(f"    {col} ({dtype})")

        observed_types = dataframe[col].dropna().map(type).unique()
        observed_type_names = {type_.__name__ for type_ in observed_types}

        column_type_info[col] = {
            "dtype": str(dtype),
            "python_types": observed_type_names,
        }

    logger.info("")

    for col, info in column_type_info.items():
        if len(info["python_types"]) > 1:
            unique_values = dataframe[col].dropna().unique().tolist()

            logger.warning(
                f"Inconsistent types in column '{col}'. "
                f"Observed types: {sorted(info['python_types'])}. "
                f"Unique values: {unique_values}"
            )
    
    return column_type_info

def summarise_info(summary_info: dict, logger: logging.Logger):
    """Summarise observed column types across all inspected datasets."""
    logger.info("SUMMARY ACROSS ALL DATASETS")
    logger.info("Observed types by column:")

    for col, info in summary_info.items():
        logger.info(f"    {col}:")
        logger.info(f"        Pandas dtypes: {sorted(info['dtypes'])}")
        logger.info(f"        Python types: {sorted(info['python_types'])}")

def inspect_data(input_dir: str, output_dir_log: str):
    logger = setup_inspection_logger(output_dir_log)

    summary_info = {}
    
    for file in Path(input_dir).rglob('*.csv'):
        logger.info(f"{file.name} has been found.")

        # Open dataset with warnings to catch potential data-inconsistencies
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            df = pd.read_csv(file, sep=";")
        for warning in caught_warnings:
            logger.warning(f"WARNING: {warning.message}")

        logger.info(f"The dataset was successfully opened.")
        
        column_type_info = extract_info(df, logger)

        for col, info in column_type_info.items():
            if col not in summary_info:
                summary_info[col] = {
                    "dtypes": set(),
                    "python_types": set(),
                }

            summary_info[col]["dtypes"].add(info["dtype"])
            summary_info[col]["python_types"].update(info["python_types"])

        logger.info(f"The evaluation of this dataset has been finished!")
        logger.info(f"______________________________________________")
        logger.info("")
    
    summarise_info(summary_info, logger)
## ________________ ##

## MAIN FUNCTION ##
def main(input_dir: str, output_dir_log: str):
    # Setting up logger
    logging.basicConfig(
        filename=output_dir_log,
        level=logging.INFO
    )

    inspect_data(input_dir, output_dir_log)

## _____________ ##

## CALL OF MAIN FUNCTION ##
if __name__ == "__main__":
    main()