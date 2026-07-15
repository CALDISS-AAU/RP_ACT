# Tsdae_Model README

This pipeline trains a sentence embedding model using **TSDAE** (Transformer-based Sequential Denoising Autoencoder).

## Overview

The pipeline:

1. Loads cleaned message data from a Parquet file.
2. Builds a SentenceTransformer model.
3. Splits the data into training, validation, and test sets.
4. Converts the training and validation data into Hugging Face datasets.
5. Trains the model using `DenoisingAutoEncoderLoss`.
6. Saves the trained model to the configured output directory.

## Directory Structure

```text
├── Data
├── Functions
│   ├── __pycache__
│   │   └── train_func.cpython-312.pyc
│   ├── data_wrangling.py
│   └── train_func.py
├── Logs
│   └── trainer_20260623_092150.log
├── Tests
├── runners
│   └── train_model.sh
├── tsdae_model_README.md
└── tsdae_model_main.py
```

## Input Data

The pipeline expects a cleaned Parquet dataset at:

```text
/ACT/Data/clean_combine/message_data.parquet
```

## Pre-processing

The pipeline relies on three functions found in:
```
/RP_ACT/Shared_Functions/modelling_functions.py
```
For the message-level model the pre-processing is kept at a small scale in order to keep each message as clean as possible.

The steps taken are as follows:
1. Splits the data into training, validation, and test sets.
    Stratification is also used on gender and age_group to keep age and gender equal in each split
2. Convertion into Hugging-face Datasets.dataset with only the text column where a 'source' and 'target' column are made. One for original text and one
3. Both steps are run through a helper function that runs both steps

The functions used are:
1. split_data
2. create_hf_dataset
3. prepare_tsdae_data

Please refer to the 'modelling_functions.py' for further instructions.

## Output

The trained model is saved to:

```text
/RP_ACT/Pipelines/Tsdae_Model/ACT_messagelvl_embed
```

Training logs are written to:

```text
/RP_ACT/Pipelines/Tsdae_Model/Logs/trainer.log
```

## Key Functions

### `_train_tsdae`

Trains a SentenceTransformer model using `DenoisingAutoEncoderLoss`.

Default training parameters:

| Parameter         | Default |
| ----------------- | ------: |
| Epochs            |       5 |
| Batch size        |      32 |
| Learning rate     |    2e-5 |
| Warmup ratio      |     0.1 |
| Max gradient norm |     1.0 |
| Save steps        |     500 |
| Logging steps     |      50 |

The trainer uses:

* `SentenceTransformerTrainingArguments`
* `SentenceTransformerTrainer`
* `DenoisingAutoEncoderLoss`
* BF16 training
* 8 dataloader workers

### `_train_message_model`

Runs the full model training workflow:

1. Builds the SentenceTransformer model.
2. Reads the input Parquet dataset.
3. Splits the data.
4. Creates Hugging Face datasets.
5. Trains and saves the TSDAE model.

## Running the Pipeline

The training script is activated through a bash script which:
1. Activates current evironment
2. Navigates to the pipeline directory
3. Defines logging directory and timestamped log file
4. Launches training schedule via Hugging Face accelerate and uv.

```bash
uv run --active accelerate launch \
 --num_processes=3 \
 --mixed_precision=bf16 \
 tsdae_model_main.py \
 2>&1 | tee "$LOGFILE"
```
## Training Configuration
Launcher | accelerate launch
Processes | 3
Mixed Precision | BF16
Package Manager | uv
Logging | Timestamped log file written to Logs/

Logs are saved as:

/RP_ACT/Pipelines/Tsdae_Model/Logs/trainer_YYYYMMDD_HHMMSS.log

where the timestamp corresponds to the start time of the training run.


The entry point calls:

```python
main()
```

which trains the message-level embedding model using the configured input, output, and logging paths.

## Configuration

The main script uses the following directory constants:

```python
INPUT_DIR_DATA = "/ACT/Data/clean_combine/message_data.parquet"
OUTPUT_DIR_LOG_TRAIN = "/RP_ACT/Pipelines/Tsdae_Model/Logs/trainer.log"
OUTPUT_DIR_MODEL = "/RP_ACT/Pipelines/Tsdae_Model/ACT_messagelvl_embed"
```

Update these paths if running the pipeline in a different environment.

## Notes

The pipeline imports shared modelling utilities from:

```python
Shared_Functions.modelling_functions
```

including:

* `build_sentence_transformer`
* `split_data`
* `create_hf_dataset`
* `prepare_tsdae_data`

Ensure these shared functions are available before running the pipeline.
