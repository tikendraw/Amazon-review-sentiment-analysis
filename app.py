import streamlit as st
import os
import tensorflow as tf
from pathlib import Path
from src import utils, data_preprocessing
import config

MODEL_PATH = Path(config.MODEL_DIR) / config.MODEL_FILENAME
VECTORIZER_PATH = Path(config.MODEL_DIR) / config.TEXT_VECTOR_FILENAME



    

def load_model_and_vectorizer(vectorizer_path, model_path):
    try:
        text_vectorizer = utils.load_text_vectorizer(vectorizer_path)
        lstm_model = tf.keras.models.load_model(model_path)
        return text_vectorizer, lstm_model
    except Exception as e:
        return None, None


def predict_sentiment(title, text, text_vectorizer, lstm_model):
    review = f'{title} {text}' # concatenate the title and text
    clean_review = data_preprocessing.clean_text(review)
    review_sequence = text_vectorizer([clean_review])
    prediction = lstm_model.predict(review_sequence)
    sentiment_score = prediction[0][0]
    sentiment_label = 'Positive' if sentiment_score >= 0.5 else 'Negative'
    return sentiment_label, sentiment_score

# Introduction and Project Information
st.title("Amazon Review Sentiment Analysis")
st.write("This is a Streamlit app for performing sentiment analysis on Amazon reviews.")
st.write("Enter the title and text of the review to analyze its sentiment.")

# User Inputs
review_title = st.text_input("Enter the review title:")
review_text = st.text_area("Enter the review text:(required)")

submit = st.button("Analyze Sentiment")



text_vectorizer, lstm_model = load_model_and_vectorizer( VECTORIZER_PATH, MODEL_PATH)
if text_vectorizer is None or lstm_model is None:
    st.error('Could not load text vectorizer and model. Aborting prediction.')

# Perform Sentiment Analysis
if submit:
    if len(review_text.strip()) > 1:        
        with st.spinner():
            sentiment_label, sentiment_score = predict_sentiment(review_title, review_text, text_vectorizer, lstm_model)
            new_sentiment_score= abs(0.5 - sentiment_score)*2

            if sentiment_score >=0.5:
                st.success(f"Sentiment: {sentiment_label} (Score: {new_sentiment_score:.2f})")
            else:
                st.error(f"Sentiment: {sentiment_label} (Score: {new_sentiment_score:.2f})")
                
                
# Project Usage and Links
st.sidebar.write("### Project Usage")
st.sidebar.write("This project performs sentiment analysis on Amazon reviews to determine whether a review's sentiment is positive or negative.")
st.sidebar.write("### GitHub Repository")
st.sidebar.write("Find the code and details on my [GitHub repository](https://github.com/tikendraw/Amazon-review-sentiment-analysis).")
st.sidebar.write("If you have any feedback or suggestions, feel free to open an issue or a pull request.")
st.sidebar.write("### Like the Project?")
st.sidebar.write("If you find this project interesting or useful, don't forget to give it a star on GitHub!")
st.sidebar.markdown("![GitHub Stars](https://img.shields.io/github/stars/tikendraw/amazon-review-sentiment-analysis)")
