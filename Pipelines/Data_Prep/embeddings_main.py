"""Main script for calculating embeddings.

To run this script, use the following command from the project root:
    uv run python -m Pipelines.Data_Prep.embeddings_main
"""

## IMPORTS ##
# Internal
from Shared_Functions.modelling_functions import calc_embeddings

# External
import os
import torch

## MAIN FUNCTION ##
def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")
    
    gpu_count = torch.cuda.device_count()

    devices = [
        f"cuda:{i}"
        for i in range(gpu_count)
    ]

    print(f"Found {gpu_count} CUDA devices")
    print(f"Using devices: {devices}")
    
    """Calculate all embeddings."""
    calc_embeddings(
        model_path='/work/RP_act/ACT/Modelling/ACT-TSDAE-messages_WoA',
        batch_size=16,
        device=devices,
        data_path="/work/RP_act/ACT/Data/Clean_Combine/message_data_without_advisors.parquet",
        filename="msg_lvl_WoA",
        normalize_embeddings=True,
    )

    calc_embeddings(
        model_path='/work/RP_act/ACT/Modelling/ACT-TSDAE-messages',
        batch_size=16,
        device=devices,
        data_path="/work/RP_act/ACT/Data/Clean_Combine/message_data.parquet",
        filename="msg_lvl_embeddings",
        normalize_embeddings=True,
    )

    calc_embeddings(
        model_path="/work/RP_act/ACT/Modelling/ACT-TSDAE-collapsed_WoA",
        batch_size=16,
        device=devices,
        data_path="/work/RP_act/ACT/Data/Clean_Combine/collapsed_messages_data_without_advisors.parquet",
        filename="collapsed_WoA_embeddings",
        normalize_embeddings=True,
    )

    calc_embeddings(
        model_path="/work/RP_act/ACT/Modelling/ACT-TSDAE-collapsed",
        batch_size=16,
        device=devices,
        data_path="/work/RP_act/ACT/Data/Clean_Combine/collapsed_messages_data.parquet",
        filename="collapsed_embeddings",
        normalize_embeddings=True,
    )

    calc_embeddings(
        model_path='BAAI/bge-m3',
        local_files_only=False,
        batch_size=16,
        device=devices,
        data_path="/work/RP_act/ACT/Data/Clean_Combine/message_data_without_advisors.parquet",
        filename="bge_m3_msg_WoA",
    )


## CALL OF MAIN FUNCTION ##
if __name__ == "__main__":
    main()