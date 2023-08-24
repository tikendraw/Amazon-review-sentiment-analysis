import os
import logging
import tensorflow as tf
from pathlib import Path
from src import utils
import config


def load_model_and_vectorizer():
    text_vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=config.MAX_TOKEN,
        output_sequence_length=config.OUTPUT_SEQUENCE_LENGTH,
        pad_to_max_tokens=True,
    )
    
    logging.info('Loading text vectorizer and model...')
    try:
        text_vectorizer = utils.load_text_vectorizer(text_vectorizer, config.TEXT_VECTOR_FILENAME)
        model_path = os.path.join(config.MODEL_DIR, config.MODEL1_FILENAME)
        lstm_model = tf.keras.models.load_model(model_path)
        return text_vectorizer, lstm_model
    except Exception as e:
        logging.error(f'Error loading vectorizer and model: {e}')
        return None, None

def predict_sentiment(title, text, text_vectorizer, lstm_model):
    review = f'{title} {text}' # concatenate the title and text
    clean_review = utils.clean_text(review)
    review_sequence = text_vectorizer([clean_review])
    prediction = lstm_model.predict(review_sequence)
    sentiment_score = prediction[0][0]
    sentiment_label = 'Positive' if sentiment_score >= 0.5 else 'Negative'
    return sentiment_label, sentiment_score

def main():
    utils.configure_logging(config.LOG_DIR, "prediction_log.txt", logging.INFO)
    text_vectorizer, lstm_model = load_model_and_vectorizer()
    
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
