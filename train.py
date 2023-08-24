import polars as pl
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src import utils, make_dataset, model
import config

import os
import logging
from pathlib import Path
    
def read_data(data_path, train_size = 0.2):
    logging.info('Reading data...')
    df = pl.read_csv(data_path)
    print('Data shape: ', df.shape)

    if train_size is None:
        sample_rate = int(df.shape[0] * 0.1)
        df = df.sample(sample_rate)
        logging.info(f'Data shape after sampling: {df.shape}')
    return df


def main():
    
    utils.configure_logging(config.LOG_DIR, "training_log.txt")  # Configure logging with log directory
    
    data_dir = Path(os.getcwd()) / 'dataset'
    data_path = data_dir / 'preprocessed_df.csv'
    model_path = Path(config.MODEL_DIR) / config.MODEL1_FILENAME
    vectorizer_path = Path(config.MODEL_DIR) / config.TEXT_VECTOR_FILENAME
    
    df = read_data(data_path, None) #config.TRAIN_SIZE)

    logging.info(f'GPU count: {len(tf.config.list_physical_devices("GPU"))}')
    
    # Text vectorization
    text_vectorizer = tf.keras.layers.TextVectorization(max_tokens = config.MAX_TOKEN,
                                                        output_sequence_length=config.OUTPUT_SEQUENCE_LENGTH,
                                                        pad_to_max_tokens=True,
                                                        )
    logging.info('Text Vectorization started...')
    
    try:
        logging.info('text_vectorizer loading weights...')
        text_vectorizer = utils.load_text_vectorizer(text_vectorizer, vectorizer_path)

    except Exception as e:
        logging.error(e)
        logging.info('Exception occured while reading text_vectorizer weights')
        logging.info('Adapting now...')
        text_vectorizer.adapt(df['review'].to_numpy())
        
    logging.info(f"text vectorizer vocab size: {text_vectorizer.vocabulary_size()}")
    logging.info('Text Vectorization done!')
    
    # Create datasets
    logging.info('Preparing dataset...')
    xtrain, xtest, ytrain, ytest = train_test_split(df['review'], df['polarity'], test_size=.05, random_state=32, stratify=df['polarity'])
    train_dataset = make_dataset.create_datasets(xtrain, ytrain, text_vectorizer, batch_size=config.BATCH_SIZE)
    test_dataset = make_dataset.create_datasets(xtest, ytest, text_vectorizer, batch_size=config.BATCH_SIZE)

    del(df, xtrain, xtest, ytrain, ytest)
    

    # Train LSTM model
    lstm_model = model.create_lstm_model(input_shape=(config.OUTPUT_SEQUENCE_LENGTH,), max_tokens=config.MAX_TOKEN, dim=config.DIM)
    lstm_model.compile(optimizer='adam',loss = tf.keras.losses.BinaryCrossentropy(), metrics=['Accuracy'])
    print(lstm_model.summary())
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=config.EARLY_STOPPING_PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(monitor='loss', filepath=model_path, save_best_only=True)
    ]
    
    try:
        lstm_model.load_weights(model_path) 
        logging.info('Model weights loaded!') 
    except Exception as e:
        logging.error('Exception occured while loading model weights', e)
        return
     
    logging.info('Model training...')
    lstm_history = lstm_model.fit(train_dataset, validation_data=test_dataset, epochs=config.EPOCHS, 
                                  steps_per_epoch=int(1.0*(len(train_dataset) / config.EPOCHS)),
                                  validation_steps=int(1.0*(len(test_dataset) / config.EPOCHS)),
                                  callbacks=callbacks)
    logging.info('Training Complete!')
    
    logging.info('Training history:')
    logging.info(lstm_history.history)
    print(pl.DataFrame(lstm_history.history))
    
    # Save text vectorizer and LSTM model
    logging.info('Saving Vectorizer and Model')
    lstm_model.save_weights(model_path, save_format='h5')
    utils.save_text_vectorizer(text_vectorizer, vectorizer_path)
    logging.info('Done')

if __name__ == "__main__":
    main()
