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
 --dataset-dir /work/RP_act/ACT/Data/Tsdae_Train_Data/conversation_seed_1849 \
 --output-dir-logs "./Pipelines/Tsdae_Model/Logs/trainer_conversations_data.log" \
 --output-dir-model "/work/RP_act/ACT/Modelling/ACT-TSDAE-conversations" \
 --epochs 4 \
 --batch-size 4 \
 --lr 1.1543077546339646e-05 \
 --warmup-ratio 0.0033987529355089694 \
 --max-grad-norm 1.0 \
 --weight-decay 0.019552921721634434 \
 2>&1 | tee "$LOGFILE"