import os
import logging
import tensorflow as tf
from pathlib import Path
from src.utils import configure_logging, load_model_and_vectorizer

from src.data_preprocessing import clean_text
import config
from tensorflow.keras.layers import TextVectorization

    
# constants
DATA_DIR = Path(os.getcwd()) / 'dataset'
DATA_PATH = DATA_DIR / 'preprocessed_df.csv'
MODEL_PATH = Path(config.MODEL_DIR) / config.MODEL_FILENAME
VECTORIZER_PATH = Path(config.MODEL_DIR) / config.TEXT_VECTOR_FILENAME
COUNTER_PATH = Path(config.MODEL_DIR) / config.COUNTER_NAME


def predict_sentiment(title, text, text_vectorizer, lstm_model):
    review = f'{title} {text}' # concatenate the title and text
    clean_review = clean_text(review)
    review_sequence = text_vectorizer([clean_review])
    prediction = lstm_model.predict(review_sequence)
    sentiment_score = prediction[0][0]
    sentiment_label = 'Positive' if sentiment_score >= 0.5 else 'Negative'
    return sentiment_label, sentiment_score

def main():
    configure_logging(config.LOG_DIR, "prediction_log.txt", logging.INFO)
    text_vectorizer, lstm_model = load_model_and_vectorizer(VECTORIZER_PATH, MODEL_PATH)
    
    if text_vectorizer is None or lstm_model is None:
        logging.error('Could not load text vectorizer and model. Aborting prediction.')
        return
    
    title = input("Enter the title of the review: ")
    text = input("Enter the text of the review: ")
    
    sentiment_label, sentiment_score = predict_sentiment(title, text, text_vectorizer, lstm_model)
    logging.debug(f'\nReview title: {title} \nReview text: {text}')
    logging.info(f'Review Sentiment: {sentiment_label} (Score: {sentiment_score:.4f})')

if __name__ == "__main__":
    main()
