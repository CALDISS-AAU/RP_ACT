## IMPORTS ##
# External
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow
# Internal
from .data_inspection import inspect_cleaned_data
## _______ ##

## HELPER FUNCTIONS ##
def _setup_cleaning_logger(output_dir_log: str) -> logging.Logger:
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

def _add_age_ranges():
    """Adds column with age group indicator
       Takes a dict of year ranges as input
    """

def _add_text_length():
    """Adds column with number of characters in msg_tekst"""

def _add_time_of_day():
    """Adds column with time of day indicator
       Takes a dict of time ranges as input
    """

def _add_season():
    """Adds column with season indicator
       Takes a dict of year ranges as input OR uses calander seasons
    """

def _add_contact_reason():
    """Adds column with contact reason from extern dataset"""

## ________________ ##

## MAIN FUNCTIONALITY ##
def add_variables(input_dir: str, output_dir_combined_data: str, output_dir_log: str):
    """Adds relevant variables to the dataset."""
    logger = _setup_cleaning_logger(output_dir_log)

        
## __________________ ##
