""" This is the main script of the ACT-project.
    This script is reserved for calling and combining pipelines from the
        Pipelines/ folders.
    For a description of the overall functionality of the individual
        pipelines, please consult their respective Pipelines/README.md.
"""
# External
import os
#Internals
from Shared_Functions.modelling_functions import (
    calc_embeddings,
    calc_mistral_embeddings,
)


def main(calculate_embeddings=False):
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
    
    

if __name__ == "__main__":
    main()
