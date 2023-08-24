# train.py
import tensorflow as tf
from sklearn.model_selection import train_test_split

def preprocess_data(df, train_size, test_size, random_state):
    
    return None

def create_datasets(x_train, y_train, text_vectorizer, batch_size):
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.map(lambda x, y: (text_vectorizer(x), y), tf.data.AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_dataset

