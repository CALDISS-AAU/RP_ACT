#!/usr/bin/env bash
set -euo pipefail

MODULE_SCRIPT="Pipelines.Data_Prep.data_prep_main"
seeds=(1013 1350 1849)

for seed in "${seeds[@]}"; do

    echo "Preparing datasets with seed $seed"

    uv run python -m "$MODULE_SCRIPT" \
        --data-path /work/RP_act/ACT/Data/Clean_Combine/message_data.parquet \
        --embedding-path /work/RP_act/ACT/Data/embeddings/msg_lvl_embeddings.npy \
        --split-seed "$seed" \
        --save-split-parquet \
        --name "msg_seed_${seed}_with_metadata"

    uv run python -m "$MODULE_SCRIPT" \
        --data-path /work/RP_act/ACT/Data/Clean_Combine/message_data_without_advisors.parquet \
        --embedding-path /work/RP_act/ACT/Data/embeddings/msg_lvl_WoA.npy \
        --split-seed "$seed" \
        --save-split-parquet \
        --name "msg_WoA_seed_${seed}_with_metadata"

    uv run python -m "$MODULE_SCRIPT" \
        --data-path /work/RP_act/ACT/Data/Clean_Combine/conversation_data.parquet \
        --embedding-path /work/RP_act/ACT/Data/embeddings/conversation_embeddings.npy \
        --split-seed "$seed" \
        --save-split-parquet \
        --name "conversation_seed_${seed}_with_metadata"

    uv run python -m "$MODULE_SCRIPT" \
        --data-path /work/RP_act/ACT/Data/Clean_Combine/conversation_data_without_advisors.parquet \
        --embedding-path /work/RP_act/ACT/Data/embeddings/conversation_WoA_embeddings.npy \
        --split-seed "$seed" \
        --save-split-parquet \
        --name "conversation_WoA_seed_${seed}_with_metadata"

    uv run python -m "$MODULE_SCRIPT" \
        --data-path /work/RP_act/ACT/Data/Clean_Combine/collapsed_messages_data.parquet \
        --embedding-path /work/RP_act/ACT/Data/embeddings/collapsed_embeddings.npy \
        --split-seed "$seed" \
        --save-split-parquet \
        --name "collapsed_seed_${seed}_with_metadata"

    uv run python -m "$MODULE_SCRIPT" \
        --data-path /work/RP_act/ACT/Data/Clean_Combine/collapsed_messages_data_without_advisors.parquet \
        --embedding-path /work/RP_act/ACT/Data/embeddings/collapsed_WoA_embeddings.npy \
        --split-seed "$seed" \
        --save-split-parquet \
        --name "collapsed_WoA_seed_${seed}_with_metadata"

done