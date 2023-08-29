import polars as pl
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src import utils, make_dataset, model
import config
import pickle
from tensorflow.keras.layers import TextVectorization
import os
import logging
from pathlib import Path
import numpy as np
from sklearn.utils import check_random_state

# constants
DATA_DIR = Path(os.getcwd()) / 'dataset'
DATA_PATH = DATA_DIR / 'preprocessed_df.csv'
MODEL_PATH = Path(config.MODEL_DIR) / config.MODEL_FILENAME
VECTORIZER_PATH = Path(config.MODEL_DIR) / config.TEXT_VECTOR_FILENAME
COUNTER_PATH = Path(config.MODEL_DIR) / config.COUNTER_NAME

def set_global_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    global random_state
    random_state = check_random_state(seed)



def read_data(DATA_PATH, train_size:float = 1.0):
    logging.info('Reading data...')
    df = pl.read_csv(DATA_PATH)
    sample_rate = int(df.shape[0] * train_size)
    df = df.sample(sample_rate, seed=config.SEED)
    logging.info(f'Data shape after sampling: {df.shape}')
    return df


def main():
    
    # Call the function to set the seeds
    set_global_seed(config.SEED)

    utils.configure_logging(config.LOG_DIR, "training_log.txt", log_level=logging.INFO)  
    
    df = read_data(DATA_PATH, config.TRAIN_SIZE)

    logging.info(f'GPU count: {len(tf.config.list_physical_devices("GPU"))}')
    
    counter = utils.load_counter(COUNTER_PATH)
    
    # Text vectorization
    logging.info('Text Vectorizer loading  ...')
    text_vectorizer = TextVectorization(max_tokens=config.MAX_TOKEN, standardize='lower_and_strip_punctuation',
                                   split='whitespace',
                                    ngrams= None ,
                                    output_mode='int',
                                    output_sequence_length=config.OUTPUT_SEQUENCE_LENGTH, 
                                    pad_to_max_tokens=True,
                                    vocabulary = list(counter.keys())[:config.MAX_TOKEN-2])

    logging.info(f"text vectorizer vocab size: {text_vectorizer.vocabulary_size()}")

    # Create datasets
    logging.info('Preparing dataset...')
    xtrain, xtest, ytrain, ytest = train_test_split(df.select('review'), df.select('polarity'), test_size=config.TEST_SIZE, random_state=config.SEED, stratify=df['polarity'])
    del(df)
    
    train_len = xtrain.shape[0]//config.BATCH_SIZE
    test_len = xtest.shape[0]//config.BATCH_SIZE
    
    logging.info('Preparing train dataset...')
    train_dataset = make_dataset.create_datasets(xtrain, ytrain, text_vectorizer, batch_size=config.BATCH_SIZE, shuffle=False)
    del(xtrain, ytrain)
    
    logging.info('Preparing test dataset...')
    test_dataset = make_dataset.create_datasets(xtest, ytest, text_vectorizer, batch_size=config.BATCH_SIZE, shuffle=False)
    del(xtest, ytest, counter, text_vectorizer )
    
    logging.info('Model loading...')
    # Train LSTM model
    lstm_model = model.create_lstm_model(input_shape=(config.OUTPUT_SEQUENCE_LENGTH,), max_tokens=config.MAX_TOKEN, dim=config.DIM)
    lstm_model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=config.LEARNING_RATE),
                       loss = tf.keras.losses.BinaryCrossentropy(), 
                       metrics=['Accuracy'])
    
    print(lstm_model.summary())
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=config.EARLY_STOPPING_PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(monitor='loss', filepath=MODEL_PATH, save_best_only=True)
    ]
    
    # Load model weights if exists
    try:
        lstm_model.load_weights(MODEL_PATH) 
        logging.info('Model weights loaded!') 
    except Exception as e:
        logging.error(f'Exception occured while loading model weights {e}')
        
     
    # Training
    logging.info('Model training...')
    lstm_history = lstm_model.fit(train_dataset, validation_data=test_dataset, epochs=config.EPOCHS, 
                                  steps_per_epoch=int(1.0*(train_len / config.EPOCHS)),
                                  validation_steps=int(1.0*(test_len / config.EPOCHS)),
                                  callbacks=callbacks)
    logging.info('Training Complete!')
    
    logging.info('Training history:')
    logging.info(lstm_history.history)
    print(pl.DataFrame(lstm_history.history))
    
    # Save text vectorizer and LSTM model
    logging.info('Saving Model')
    lstm_model.save(MODEL_PATH, save_format='h5')
    logging.info('Done')

if __name__ == "__main__":
    main()
