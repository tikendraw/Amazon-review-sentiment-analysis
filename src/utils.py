import logging
import os
import pickle
import re
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

from .data_preprocessing import clean_text


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
    return
    

def save_text_vectorizer(text_vectorizer, filename):
    config = text_vectorizer.get_config()
    with open(filename, 'wb') as f:
        pickle.dump({'config': config}, f)


def load_counter(filename):
    with open (filename,'rb')  as counter :
        return pickle.load(counter)
    

def load_model(model, model_dir):
    """Load the model from disk."""
    # Load the Keras model
    return model.load_weights(model_dir)


def load_text_vectorizer(vectorizer_path):
    from_disk = pickle.load(open(vectorizer_path, "rb"))
    return TextVectorization.from_config(from_disk['config'])


def load_model_and_vectorizer(vectorizer_path, model_path):
    try:
        text_vectorizer = load_text_vectorizer(vectorizer_path)
        lstm_model = tf.keras.models.load_model(model_path)
        return text_vectorizer, lstm_model
    except Exception as e:
        logging.error(f'Error loading vectorizer and model: {e}')
        return None, None



def predict_sentiment(title, text, text_vectorizer, lstm_model):
    review = f'{title} {text}' # concatenate the title and text
    clean_review = clean_text(review)
    review_sequence = text_vectorizer([clean_review])
    prediction = lstm_model.predict(review_sequence)
    sentiment_score = prediction[0][0]
    sentiment_label = 'Positive' if sentiment_score >= 0.5 else 'Negative'
    return sentiment_label, sentiment_score
