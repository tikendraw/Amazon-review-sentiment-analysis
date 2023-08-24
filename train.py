import os
import logging
import polars as pl
import tensorflow as tf
from pathlib import Path
from src import utils, make_dataset, model
import config

# Configure logging
def configure_logging(log_dir):
    log_file = os.path.join(log_dir, "training_log.txt")
    
    # Configure logging to both console and file
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.StreamHandler(),
                            logging.FileHandler(log_file)
                        ])
    
def read_data(data_path, train_size = 0.2):
    logging.info('Reading data...')
    df = pl.read_csv(data_path)
    sample_rate = int(df.shape[0] * 0.1)
    df = df.sample(sample_rate)
    logging.info(f'Data shape after sampling: {df.shape}')
    return df

def main():
    
    configure_logging(config.LOG_DIR)  # Configure logging with log directory
    
    data_dir = Path(os.getcwd()) / 'dataset'
    data_path = data_dir / 'preprocessed_df.csv'
    
    df = read_data(data_path, config.TRAIN_SIZE)

    logging.info(f'GPU count: {len(tf.config.list_physical_devices("GPU"))}')
    
    # Text vectorization
    text_vectorizer = utils.get_text_vectorizer(config.TEXT_VECTOR_FILENAME)
    logging.info('Text Vectorization adapting...')
    # text_vectorizer.adapt(df['review'].to_numpy())
    text_vectorizer = utils.load_text_vectorizer(text_vectorizer, config.TEXT_VECTOR_FILENAME)
    logging.info('Text Vectorization done!')
    
    # Create datasets
    logging.info('Preparing dataset...')
    train_dataset = make_dataset.create_datasets(df['review'], df['polarity'], text_vectorizer, batch_size=config.BATCH_SIZE)
    del(df)
    
    for i, j in train_dataset.take(1):
        print(tf.shape(i))
        break
    
    # Train LSTM model
    lstm_model = model.create_lstm_model(input_shape=(config.OUTPUT_SEQUENCE_LENGTH,), max_tokens=config.MAX_TOKEN, dim=config.DIM)
    lstm_model.compile(optimizer='adam',loss = tf.keras.losses.BinaryCrossentropy(), metrics=['Accuracy'])
    print(lstm_model.summary())
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=config.EARLY_STOPPING_PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(monitor='loss', filepath=os.path.join(config.MODEL_DIR, config.MODEL1_FILENAME), save_best_only=True)
    ]
    
    logging.info('Model training...')
    lstm_history = lstm_model.fit(train_dataset, epochs=config.EPOCHS, steps_per_epoch=int(0.1*(len(train_dataset) / config.EPOCHS)), callbacks=callbacks)
    logging.info('Training Complete!')
    
    logging.info('Training history:')
    logging.info(lstm_history.history)
    print(pl.DataFrame(lstm_history.history))
    
    # Save text vectorizer and LSTM model
    logging.info('Saving Vectorizer and Model')
    utils.save_text_vectorizer(text_vectorizer, config.TEXT_VECTOR_FILENAME)
    lstm_model.save(os.path.join(config.MODEL_DIR, config.MODEL1_FILENAME))
    logging.info('Done')

if __name__ == "__main__":
    main()