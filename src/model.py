# model.py
from tensorflow import keras
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Embedding,
                                     TextVectorization)


def create_lstm_model(input_shape, max_tokens, dim):
    inputs = keras.Input(shape=(input_shape))
    embedding_layer = Embedding(input_dim=max_tokens, output_dim=dim, mask_zero=True, input_length=input_shape, name='embedding_layer')(inputs)
    x = LSTM(16, return_sequences=True, name = 'lstm_layer_1')(embedding_layer)
    x = LSTM(16, name = 'lstm_layer_2')(x)
    x = Dropout(0.4, name ='dropout_layer')(x)
    x = Dense(64, activation='relu', name = 'dense_layer_1')(x)
    outputs = Dense(1, activation='sigmoid', name = 'dense_layer_2_final')(x)
    return keras.Model(inputs=inputs, outputs=outputs, name='model_lstm')

