#!/bin/bash
set -e

cd "/work/JosefineMarianneChristensen#0406/RP_ACT"
source ".venv/bin/activate"

export PYTHONPATH="$PYTHONPATH:./Pipelines/Tsdae_Model"


LOG_DIR='./Pipelines/Tsdae_Model/Logs'
LOGFILE="$LOG_DIR/trainer_$(date +%Y%m%d_%H%M%S).log"

uv run --active accelerate launch \
 --num_processes=4 \
 --num_machines=1 \
 --dynamo_backend=no \
 --mixed_precision=bf16 \
 Pipelines/Tsdae_Model/tsdae_model_main.py \
 --data-path /work/RP_act/ACT/Data/Clean_Combine/message_data.parquet \
 --output-dir-logs "./Pipelines/Tsdae_Model/Logs/trainer_message_data.log" \
 --output-dir-model "/work/RP_act/ACT/Modelling/ACT-TSDAE-message-v1" \
 --epochs 6 \
 --batch-size 32 \
 --lr 2.9006604711101802e-06 \
 --warmup-ratio 0.029801408130959731 \
 --max-grad-norm 2.0 \
 --weight-decay 0.0011808351892388493 \
 2>&1 | tee "$LOGFILE"