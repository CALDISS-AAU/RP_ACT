import pandas as pd
import numpy as np

train = pd.read_parquet('/work/RP_act/ACT/Data/splits/seed_958/train.parquet')
test = pd.read_parquet('/work/RP_act/ACT/Data/splits/seed_958/test.parquet')
val = pd.read_parquet('/work/RP_act/ACT/Data/splits/seed_958/val.parquet')


# Check whether conversations overlap
train_conversations = set(train["conversation_code"])
val_conversations = set(val["conversation_code"])
test_conversations = set(test["conversation_code"])

print("Train / Val overlap:",
      len(train_conversations & val_conversations))

print("Train / Test overlap:",
      len(train_conversations & test_conversations))

print("Val / Test overlap:",
      len(val_conversations & test_conversations))


