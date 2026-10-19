#!/bin/bash
set -euo pipefail

cd "/work/JosefineMarianneChristensen#0406/RP_ACT"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

LOG_DIR="$PWD/Pipelines/Hpo/from_runs/collapsed_message_level_WoA_hpo/Logs"
OUTPUT_DIR="$PWD/Pipelines/Hpo/from_runs/output_collapsed_messages_WoA"
DATASET_DIR="/work/RP_act/ACT/Data/Tsdae_Train_Data/collapsed_no_advisor_seed_1849"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="$i" .venv/bin/python Pipelines/Hpo/hpo_main.py \
        --dataset-dir "$DATASET_DIR" \
        --study_name "collapsed_messages_WoA_small_batch_gpu" \
        --n-trials 5 \
        --output-dir-logs "$LOG_DIR" \
        --output-trial-dir "$OUTPUT_DIR" \
        > "$LOG_DIR/hpo_${TIMESTAMP}_gpu_${i}.log" 2>&1 &
done

wait