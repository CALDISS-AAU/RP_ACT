#!/bin/bash
set -e

cd "/work/JosefineMarianneChristensen#0406/RP_ACT"
source ".venv/bin/activate"

export PYTHONPATH="$PYTHONPATH:./Pipelines/Tsdae_Model"


LOG_DIR='./Pipelines/Tsdae_Model/Logs'
LOGFILE="$LOG_DIR/messages_WoA_trainer_$(date +%Y%m%d_%H%M%S).log"

uv run --active accelerate launch \
 --num_processes=4 \
 --num_machines=1 \
 --dynamo_backend=no \
 --mixed_precision=bf16 \
 Pipelines/Tsdae_Model/tsdae_model_main.py \
 --dataset-dir /work/RP_act/ACT/Data/Tsdae_Train_Data/msg_no_advisor_seed_1849 \
 --output-dir-logs "./Pipelines/Tsdae_Model/Logs/trainer_message_WoA_data.log" \
 --output-dir-model "/work/RP_act/ACT/Modelling/ACT-TSDAE-messages_WoA" \
 --epochs 5 \
 --batch-size 8 \
 --lr 1.3292210351240338e-05 \
 --warmup-ratio 0.08404404487865975 \
 --max-grad-norm 2.0 \
 --weight-decay 0.06951877405651552 \
 2>&1 | tee "$LOGFILE"