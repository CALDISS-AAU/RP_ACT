# Baseline + other test model
PYTHONPATH=. uv run /work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/validation/InformationRetrievalEval_main.py \
    --model-path BAAI/bge-m3 \
    --no-local-files-only \
    --dataset-path /work/RP_act/ACT/Data/Tsdae_Train_Data/msg_no_advisor_seed_1849 \
    --output-path /work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/validation/Results/IRE_results/bge_m3_msg_woa_1849 \
    --name bge_m3_msg_woa_1849_eval

PYTHONPATH=. uv run /work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/validation/InformationRetrievalEval_main.py \
    --model-path intfloat/multilingual-e5-large \
    --no-local-files-only \
    --dataset-path /work/RP_act/ACT/Data/Tsdae_Train_Data/msg_no_advisor_seed_1849 \
    --output-path /work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/validation/Results/IRE_results/e5_large_msg_woa_1849 \
    --name e5_large_msg_woa_1849_eval
# Collapsed
PYTHONPATH=. uv run /work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/validation/InformationRetrievalEval_main.py \
    --model-path /work/RP_act/ACT/Modelling/ACT-TSDAE-collapsed_WoA \
    --local-files-only \
    --dataset-path /work/RP_act/ACT/Data/Tsdae_Train_Data/collapsed_no_advisor_seed_1849 \
    --output-path /work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/validation/Results/IRE_results/collapsed_woa_1849 \
    --name collapsed_WoA_1849_eval

PYTHONPATH=. uv run /work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/validation/InformationRetrievalEval_main.py \
    --model-path /work/RP_act/ACT/Modelling/ACT-TSDAE-collapsed \
    --local-files-only \
    --dataset-path /work/RP_act/ACT/Data/Tsdae_Train_Data/collapsed_seed_1849 \
    --output-path /work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/validation/Results/IRE_results/collapsed_1849 \
    --name collapsed_1849_eval

# Messge
PYTHONPATH=. uv run /work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/validation/InformationRetrievalEval_main.py \
    --model-path /work/RP_act/ACT/Modelling/ACT-TSDAE-messages \
    --local-files-only \
    --dataset-path /work/RP_act/ACT/Data/Tsdae_Train_Data/msg_seed_1849 \
    --output-path /work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/validation/Results/IRE_results/message_1849 \
    --name message_1849_eval

PYTHONPATH=. uv run /work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/validation/InformationRetrievalEval_main.py \
    --model-path /work/RP_act/ACT/Modelling/ACT-TSDAE-messages_WoA \
    --local-files-only \
    --dataset-path /work/RP_act/ACT/Data/Tsdae_Train_Data/msg_no_advisor_seed_1849 \
    --output-path /work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/validation/Results/IRE_results/message_1849 \
    --name message_WoA_1849_eval

# Conversation
PYTHONPATH=. uv run /work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/validation/InformationRetrievalEval_main.py \
    --model-path /work/RP_act/ACT/Modelling/ACT-TSDAE-conversations_WoA \
    --local-files-only \
    --dataset-path /work/RP_act/ACT/Data/Tsdae_Train_Data/conversation_no_advisor_seed_1849 \
    --output-path /work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/validation/Results/IRE_results/conversation_WoA_1849 \
    --name conversation_WoA_1849_eval

PYTHONPATH=. uv run /work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/validation/InformationRetrievalEval_main.py \
    --model-path /work/RP_act/ACT/Modelling/ACT-TSDAE-conversations \
    --local-files-only \
    --dataset-path /work/RP_act/ACT/Data/Tsdae_Train_Data/conversation_seed_1849 \
    --output-path /work/MatiasKokholmAppel#3073/RP_ACT/Pipelines/validation/Results/IRE_results/conversation_WoA_1849 \
    --name conversation_1849_eval