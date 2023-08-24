import os
from pathlib import Path

# Data paths
DATA_DIR = Path(os.getcwd()) / "dataset"
PREPROCESSED_DATA_PATH = Path(DATA_DIR)/ "preprocessed_df.csv"

# Model paths
MODEL_DIR = Path(os.getcwd())/ "model"
                 
TEXT_VECTOR_FILENAME = "tv_layer.pkl"
MODEL1_FILENAME = "model1.h5"

# Model hyperparameters
MAX_TOKEN = 100_000
OUTPUT_SEQUENCE_LENGTH = 100
BATCH_SIZE = 64
DIM = 8
EPOCHS = 10