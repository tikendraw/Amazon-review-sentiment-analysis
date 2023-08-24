import os
from pathlib import Path

# Data paths
DATA_DIR = Path(os.getcwd()) / "dataset"
PREPROCESSED_DATA_PATH = Path(DATA_DIR)/ "preprocessed_df.csv"

# Model paths
MODEL_DIR = Path(os.getcwd())/ "model"
                 
TEXT_VECTOR_FILENAME = "model/tv_layer.pkl"
MODEL1_FILENAME = "model1.h5"

# Model hyperparameters
MAX_TOKEN = 100_000  # don't change this
OUTPUT_SEQUENCE_LENGTH = 175 # don't change this
BATCH_SIZE = 64
DIM = 8
EPOCHS = 2
TRAIN_SIZE = 0.1

#callback
EARLY_STOPPING_PATIENCE = 2