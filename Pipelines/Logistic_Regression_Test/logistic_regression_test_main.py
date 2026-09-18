"""Main script for the Logistic_Regression_Test pipeline."""
# External
import pandas as pd
import os
# Internal
from Functions.logistic_reg_func import _create_report, _logistic_regression, _prep_data
#Internals
from Shared_Functions.modelling_functions import (
    calc_embeddings,
    calc_mistral_embeddings,
    split_data
)

def main(calculate_embeddings=False):

    # Embedding stuff
    if calculate_embeddings:
        calc_embeddings(
            filename="msg_lvl_unnormalized",
            normalize_embeddings=False,
            )
        
        calc_embeddings()

        calc_embeddings(
            model_path='/work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/Tsdae_Model/modelling/ACT-TSDAE-Collapsed-v2',
            data_path='/work/RP_act/ACT/Data/Clean_Combine/collapsed_messages_data.parquet',
            filename="collapsed_normalised"
        )

        calc_embeddings(
            model_path="BAAI/bge-m3",
            filename="bge_m3_baseline_embeddings",
            local_files_only=False,
            )
    
    
    calc_mistral_embeddings()

    # LR Stuff
    df = pd.read_parquet('/work/RP_act/ACT/Data/Clean_Combine/message_data.parquet')

    df_model, val_df, test_df = _prep_data(df=df)

    _, report = _logistic_regression(df=df_model, embedding_path='/work/RP_act/ACT/Data/embeddings/msg_lvl_embeddings.npy', filename= 'message_norm_cm')

    _create_report(
    report,
    filename="message_norm_LR.xlsx"
)

if __name__ == "__main__":
    main()