"""Main script for the Logistic_Regression_Test pipeline."""
# External
import  pandas as pd
# Internal
from Functions.logistic_reg_func import (
    _prep_data,
    _logistic_regression,
    _create_report
)

def main():
    df = pd.read_parquet('/work/RP_act/ACT/Data/Clean_Combine/message_data.parquet')

    df_model = _prep_data(df=df)

    _, report = _logistic_regression(df=df_model)

    _create_report(
    report,
    filename="tsdae_gender_normalized.xlsx"
)

if __name__ == "__main__":
    main()