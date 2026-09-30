#!/bin/bash
set -euo pipefail

cd "/work/JosefineMarianneChristensen#0406/RP_ACT" || exit 1
source ".venv/bin/activate"

export PYTHONPATH="/work/JosefineMarianneChristensen#0406/RP_ACT:${PYTHONPATH:-}"

LOG_DIR="/work/JosefineMarianneChristensen#0406/RP_ACT/Pipelines/Hpo/from_runs/message_level_WoA_hpo/Logs"
# LOGFILE="$LOG_DIR/hpo_$(date +%Y%m%d_%H%M%S).log"
OUTPUT_DIR="/work/JosefineMarianneChristensen#0406/RP_ACT/Pipelines/Hpo/from_runs/output_messages_WoA"
DATA_PATH="/work/RP_act/ACT/Data/Clean_Combine/message_data_without_advisors.parquet"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOGFILE_GPU0="$LOG_DIR/hpo_gpu0_${TIMESTAMP}.log"
LOGFILE_GPU1="$LOG_DIR/hpo_gpu1_${TIMESTAMP}.log"

CUDA_VISIBLE_DEVICES=0 uv run Pipelines/Hpo/hpo_main.py \
  --data-path "/work/RP_act/ACT/Data/Clean_Combine/message_data_without_advisors.parquet" \
  --prepare-data \
  --output-dir-logs "/work/JosefineMarianneChristensen#0406/RP_ACT/Pipelines/Hpo/from_runs/message_level_WoA_hpo/Logs" \
  --output-trial-dir "/work/JosefineMarianneChristensen#0406/RP_ACT/Pipelines/Hpo/from_runs/output_messages_WoA" \
  2>&1 | tee "$LOGFILE_GPU0" &

PID_GPU0=$!

CUDA_VISIBLE_DEVICES=1 uv run Pipelines/Hpo/hpo_main.py \
  --data-path "/work/RP_act/ACT/Data/Clean_Combine/message_data_without_advisors.parquet" \
  --prepare-data \
  --output-dir-logs "/work/JosefineMarianneChristensen#0406/RP_ACT/Pipelines/Hpo/from_runs/message_level_WoA_hpo/Logs" \
  --output-trial-dir "/work/JosefineMarianneChristensen#0406/RP_ACT/Pipelines/Hpo/from_runs/output_messages_WoA" \
  2>&1 | tee "$LOGFILE_GPU1" &

PID_GPU1=$!


wait "$PID_GPU0"
wait "$PID_GPU1"
