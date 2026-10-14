#!/bin/bash
set -euo pipefail

cd "/work/JosefineMarianneChristensen#0406/RP_ACT" || exit 1
source ".venv/bin/activate"

export PYTHONPATH="/work/JosefineMarianneChristensen#0406/RP_ACT:${PYTHONPATH:-}"

LOG_DIR="/work/JosefineMarianneChristensen#0406/RP_ACT/Pipelines/Hpo/from_runs/conversation_level_hpo/Logs"
# LOGFILE="$LOG_DIR/hpo_$(date +%Y%m%d_%H%M%S).log"
OUTPUT_DIR="/work/JosefineMarianneChristensen#0406/RP_ACT/Pipelines/Hpo/from_runs/output_conversations"
DATA_PATH="/work/RP_act/ACT/Data/Clean_Combine/conversation_data.parquet"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOGFILE="$LOG_DIR/hpo_${TIMESTAMP}.log"

uv run Pipelines/Hpo/hpo_main.py \
  --dataset-dir "/work/RP_act/ACT/Data/Tsdae_Train_Data/conversation_seed_1849" \
  --max_seq_length 1024 \
  --study_name "conversations_small_batch_gpu" \
  --output-dir-logs "/work/JosefineMarianneChristensen#0406/RP_ACT/Pipelines/Hpo/from_runs/conversation_level_hpo/Logs" \
  --output-trial-dir "/work/JosefineMarianneChristensen#0406/RP_ACT/Pipelines/Hpo/from_runs/output_conversations" \
  2>&1 | tee "$LOGFILE"
