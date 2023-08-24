import re
import pickle
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf
import logging
import os


# Configure logging
def configure_logging(log_dir, log_filename, log_level=logging.INFO):
    
    log_dir = Path(log_dir).mkdir(exist_ok=True)
    log_file = log_dir / log_filename
    
    # Configure logging to both console and file
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.StreamHandler(),
                            logging.FileHandler(log_file)
                        ])
    
def clean_text(x):
    x = re.sub(r'[^\w\s]', '', x)
    x = x.lower()
    return x

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


# src/data_preprocessing.py

import re
import polars as pl
from pathlib import Path
from sklearn.model_selection import train_test_split

def preprocess_data(data_dir):
    # Read the CSV file using Polars
    df = pl.read_csv(data_dir / 'train.csv', new_columns=['polarity', 'title', 'text'])

    assert df['polarity'].max()==2
    assert df['polarity'].min()==1


    # Drop rows with null values
    df.drop_nulls()

    # Map polarity to binary values (0 for negative, 1 for positive)
    df = df.with_columns([
        pl.col('polarity').apply(lambda x: 0 if x == 1 else 1)
    ])

    # Cast polarity column to Int16
    df = df.with_columns([
        pl.col('polarity').cast(pl.Int16, strict=False)
    ])

    # Combine title and text columns to create the review column
    df = df.with_columns([
        (pl.col('title') + ' ' + pl.col('text')).alias('review')
    ])
    
    df = df.with_columns([
	(pl.col('review').str().lower())
	])
	
    # Select relevant columns
    df = df.select(['review', 'polarity'])

    # Perform text cleaning using a function
    df = df.with_columns([
        pl.col('review').apply(clean_text)
    ])

    df.write_csv(data_dir/'preprocessed_df.csv')
    

def clean_text(x: str) -> str:
    # Remove punctuation and convert to lowercase
    x = re.sub(r'[^\w\s]', '', x)
    x = x.lower()
    return x
