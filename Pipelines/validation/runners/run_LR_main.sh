#!/bin/bash

SPLIT_ROOT="/work/RP_act/ACT/Data/splits"
LR_SCRIPT="/work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/validation/logistic_regression_test_main.py"
# All dataset types
datasets=(
    "message_data"
    "message_data_without_advisors"
    "conversation_data"
    "conversation_data_without_advisors"
    "collapsed_messages_data"
    "collapsed_messages_data_without_advisors"
)
seeds=(1013 1350 1849)

# for seed in 793,958,1013,1219,1350,1849; do
for dataset in "${datasets[@]}"; do
    for seed in "${seeds[@]}"; do

    echo "Evaluating Logistic Regression with $seed $data"
            PYTHONPATH=. uv run "$LR_SCRIPT" \
                --split-dir "$SPLIT_ROOT/$dataset/seed_$seed" \
                --seed "$seed" \
                --C 1.0 \
                --class-weight balanced \
                --filename "LR_${dataset}_seed_${seed}"
    done
done