#!/bin/bash
set -e

cd "/work/JosefineMarianneChristensen#0406/RP_ACT"
source ".venv/bin/activate"

export PYTHONPATH="$PYTHONPATH:./Pipelines/Tsdae_Model"


LOG_DIR='./Pipelines/Tsdae_Model/Logs'
LOGFILE="$LOG_DIR/conversations_trainer_$(date +%Y%m%d_%H%M%S).log"

uv run --active accelerate launch \
 --num_processes=4 \
 --num_machines=1 \
 --dynamo_backend=no \
 --mixed_precision=bf16 \
 Pipelines/Tsdae_Model/tsdae_model_main.py \
 --data-path /work/RP_act/ACT/Data/Clean_Combine/conversation_data.parquet \
 --output-dir-logs "./Pipelines/Tsdae_Model/Logs/trainer_conversations_data.log" \
 --output-dir-model "/work/RP_act/ACT/Modelling/ACT-TSDAE-conversations" \
 --epochs 4 \
 --batch-size 4 \
 --lr 1.4916427128440363e-06 \
 --warmup-ratio 0.138193443011922 \
 --max-grad-norm 1.0 \
 --weight-decay 0.0438248914201757 \
 2>&1 | tee "$LOGFILE"