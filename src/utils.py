import re
import pickle
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf
import logging
import os
from pathlib import Path

# Configure logging
def configure_logging(log_dir, log_filename, log_level=logging.INFO):
    
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / log_filename
    
    # Configure logging to both console and file
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.StreamHandler(),
                            logging.FileHandler(log_file)
                        ])
    

def save_text_vectorizer(text_vectorizer, filename):
    config = text_vectorizer.get_config()
    weights = text_vectorizer.get_weights()
    with open(filename, 'wb') as f:
        pickle.dump({'config': config, 'weights': weights}, f)

def load_text_vectorizer(text_vectorizer, filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        text_vectorizer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
        text_vectorizer.set_weights(data['weights'])
        return text_vectorizer

