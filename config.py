import os
from pathlib import Path

cur_dir= Path(os.getcwd())

# Data paths
DATA_DIR = cur_dir / "dataset"
PREPROCESSED_DATA_PATH = DATA_DIR / "preprocessed_df.csv"

# Paths
MODEL_DIR = cur_dir / "model"
LOG_DIR = cur_dir / "logs"
VECTORIZE_PATH = cur_dir / 'model' / 'text_vectorizer.pkl'

TEXT_VECTOR_FILENAME = "text_vectorizer.pkl"
MODEL_FILENAME = "full_model.h5"
COUNTER_NAME  = "counter.pkl"

# Text Vectorizer hyperparameters
MAX_TOKEN = 100_000  # don't change this
OUTPUT_SEQUENCE_LENGTH = 175 # don't change this

# Model hyperparameters
BATCH_SIZE = 32
DIM = 8
EPOCHS = 10
TRAIN_SIZE = 0.05
TEST_SIZE = 0.01
LEARNING_RATE = 0.002
RANDOM_STATE = 42
SEED = 42
#callback
EARLY_STOPPING_PATIENCE = 2
