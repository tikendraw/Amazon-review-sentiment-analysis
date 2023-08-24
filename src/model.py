# model.py
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, TextVectorization
from tensorflow import keras

def create_lstm_model(input_shape, max_tokens, dim):
    inputs = keras.Input(shape=input_shape)
    embedding_layer = Embedding(input_dim=max_tokens, output_dim=dim, mask_zero=True, input_length=input_shape[0])(inputs)
    x = LSTM(16, return_sequences=True)(embedding_layer)
    x = LSTM(16)(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return keras.Model(inputs=inputs, outputs=outputs, name='model_lstm')

