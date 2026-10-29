#!/usr/bin/env bash
set -euo pipefail

SCRIPT="/work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/validation/InformationRetrievalEval_main.py"
datasets=(
    '/work/RP_act/ACT/Data/Tsdae_Train_Data/collapsed_seed_1013_with_metadata'
    '/work/RP_act/ACT/Data/Tsdae_Train_Data/collapsed_WoA_seed_1013_with_metadata'
    '/work/RP_act/ACT/Data/Tsdae_Train_Data/conversation_seed_1013_with_metadata'
    '/work/RP_act/ACT/Data/Tsdae_Train_Data/conversation_WoA_seed_1013_with_metadata'
    '/work/RP_act/ACT/Data/Tsdae_Train_Data/msg_seed_1013_with_metadata'
    '/work/RP_act/ACT/Data/Tsdae_Train_Data/msg_WoA_seed_1013_with_metadata'
)
models=(
    '/work/RP_act/ACT/Modelling/ACT-TSDAE-collapsed'
    '/work/RP_act/ACT/Modelling/ACT-TSDAE-collapsed_WoA'
    '/work/RP_act/ACT/Modelling/ACT-TSDAE-conversations_WoA'
    '/work/RP_act/ACT/Modelling/ACT-TSDAE-messages'
    '/work/RP_act/ACT/Modelling/ACT-TSDAE-messages_WoA'
)

for dataset in "${datasets[@]}"; do
    dataset_name=$(basename "$dataset")

    for model in "${models[@]}"; do
        echo "using $model on $dataset"
        model_name=$(basename "$model")

        output_name="IRE_${dataset_name}_${model_name}"

        echo "Evaluating $model_name on $dataset_name"
        echo "Output name: $output_name"

        PYTHONPATH="/work/MatiasKokholmAppel#3073/RP_ACT${PYTHONPATH:+:$PYTHONPATH}" uv run "$SCRIPT" \
            --model-path "$model" \
            --local-files-only \
            --dataset-path "$dataset" \
            --output-path "/work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/validation/Results/IRE_results/Cross_Test/$output_name" \
            --name "$output_name"
    done
done