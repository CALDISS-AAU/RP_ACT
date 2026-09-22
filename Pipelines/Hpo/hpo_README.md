# Hpo README
This pipeline trains and evaluates a **TSDAE** model using **Hyperparameter Optimization** using optuna.

## Overview

The pipeline:
1. Initiates a a fresh model for every optuna trial
2. Initiates a loss function (Denoising Auto Encoder Loss)
3. Creates an Optuna hyperparameter search space.
4. Defines the HPO objective.
5. Runs the trial on two parrallel workers for 5 trials


## Directory Structure

```text
.
├── Data
├── Functions
│   ├── __pycache__
│   │   └── hpo_func.cpython-312.pyc
│   └── hpo_func.py
├── Logs
│   ├── hpo_20260703_145809.log
│   ├── hpo_gpu0_20260710_120808.log
│   ├── hpo_gpu1_20260710_120808.log
│   └── tsdae_hpo_journal.log
├── Tests
├── hpo_README.md
├── hpo_main.py
├── output
└── runners
    └── run_hpo.sh
````

## Input Data

The pipeline expects a cleaned Parquet dataset at:
```text
/ACT/Data/clean_combine/message_data.parquet
```

## Training
The Pipeline trains a model based on the defined parameters from the search space interatively. The HPO evaluates on the validation dataset and the loss value thus choosing the best parameters that produces the lowest loss.

The search space is currently defined as:
| Hyperparameter                | Search Space            | Type                |
| ----------------------------- | ----------------------- | ------------------- |
| `num_train_epochs`            | `3–6`                   | Integer             |
| `per_device_train_batch_size` | `[16, 32]`              | Categorical         |
| `warmup_ratio`                | `0.0–0.15`              | Float               |
| `learning_rate`               | `1e-6–5e-5` (log scale) | Float (log-uniform) |
| `weight_decay`                | `0.0–0.1`               | Float               |
| `max_grad_norm`               | `[0.5, 1.0, 2.0]`       | Categorical         |

## Output
The results are saved to an Optuna JournalStorage and most importantly the log files:
> The HPO training outputs two files from each parrallel run: hpo_gpu{ID}_{TIMESTAMP}.log

Consult the log files to inspect training schedule, how it behaves and best trial.

### Key Functions 

### `_run_tsdae_hpo`
Runs the HPO trial using a newly initiated SentenceTransfomer using `DenoisingAutoEncoderLoss`.

For standards the trial initiates `BAAI/bge-m3` with a max sequence length of 128 and `mean pooling`.
The model trains for 5 trials as a standard.

The trainer uses:
* `SentenceTransformerTrainingArguments`
* `SentenceTransformerTrainer`
* `DenoisingAutoEncoderLoss`
* BF16 training
* 8 dataloader workers

### Training Configuration
The HPO is initiated twice using GPU ID's. Make sure you have two GPU's available when starting or else the script will fail.


CUDA_VISIBLE_DEVICES=0 uv run Pipelines/Hpo/hpo_main.py \
  --data-path "/work/RP_act/ACT/Data/Clean_Combine/collapsed_messages_data.parquet" \
  --output-dir-logs "./RP_ACT/Pipelines/Hpo/Logs" \
  --output-trial-dir "./RP_ACT/Pipelines/Hpo/output" \
  2>&1 | tee "$LOGFILE_GPU0" &

The entry point calls:

```python
main()
```
Which starts the HPO study using the configured input, output and logging paths.

### Configuration

The main script has modular paths for input, output and logging but takes the RP_ACT repo as standard:

```python
OUTPUT_DIR="./RP_ACT/Pipelines/Hpo/output"
DATA_PATH="/work/RP_act/ACT/Data/Clean_Combine/collapsed_messages_data.parquet"
```
Update these paths if running the pipeline in a different environment.

