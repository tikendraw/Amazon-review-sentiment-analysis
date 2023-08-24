import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tensorflow.keras.layers import Embedding ,LSTM, Dense, Dropout, TextVectorization
import polars as pl
import os, re
import pandas as pd
from pathlib import Path
import pickle
from funcyou.sklearn.metrics import calculate_results
from funcyou.plot import plot_history

data_dir = Path(os.getcwd()).parent /'dataset'
data_dir

#reading data
df = pl.read_csv(data_dir/'train.csv',new_columns = ['polarity', 'title','text'])  # gives TextFileReader, which is iterable with chunks of 1000 rows.
#drop nulls
df.drop_nulls()
# We will map the polarity between 0 for negative sentiment to 1 for positive sentiment
df = df.with_columns([
                    pl.col('polarity').apply(lambda x: 0 if x == 1 else 1).alias('polarity')
                     ])

df = df.with_columns([
                     pl.col('polarity').cast(pl.Int16, strict=False).alias('polarity')
                     ])

# Note: We will be combining text and title columns . makes more sense.
df = df.with_columns([
    (pl.col('title')+' ' + pl.col('text')).alias('review')
])


# %%
df = df.select(['review','polarity'])

# %%
def clean_text(x: str) -> str:
    x = re.sub(r'[^\w\s]', '', x)  # Remove punctuation
    x = x.lower()  # Convert to lowercase
    return x    

pattern = r'\b\w+\b'

# dff.write_csv(data_dir / 'preprocessed_df.csv')

df = pl.read_csv(data_dir/'preprocessed_df.csv')

# Data Preparation
xtrain, xtest, ytrain, ytest = train_test_split( df.select('review'), df.select('polarity'),train_size=0.1, test_size=  .001,  random_state = 89)
del(df) # deleting variables to keep the memory free

#train
train_feature = tf.data.Dataset.from_tensor_slices(xtrain.to_numpy())
train_label = tf.data.Dataset.from_tensor_slices(ytrain.to_numpy())
#test
test_feature = tf.data.Dataset.from_tensor_slices(xtest.to_numpy())
test_label = tf.data.Dataset.from_tensor_slices(ytest.to_numpy())

BATCH_SIZE = 64

# Text vectorization
MAX_TOKEN = 100_000
OUTPUT_SEQUENCE_LENGTH = 100  # limiting reviews to 200 words
text_vectorizer = TextVectorization(max_tokens=MAX_TOKEN, standardize='lower_and_strip_punctuation',
                                   split='whitespace',
                                    ngrams= None ,
                                    output_mode='int',
                                    output_sequence_length=OUTPUT_SEQUENCE_LENGTH, 
                                    pad_to_max_tokens=False)


text_vector_filename = 'tv_layer.pkl'

# adapt the text_vectorizer to the data

# Pickle the config and weights
pickle.dump({'config': text_vectorizer.get_config(),
             'weights': text_vectorizer.get_weights()}
            , open(text_vector_filename, "wb"))


# `weights` to load the trained weights. 
from_disk = pickle.load(open(text_vector_filename, "rb"))
text_vectorizer = TextVectorization.from_config(from_disk['config'])
# You have to call `adapt` with some dummy data (BUG in Keras)
text_vectorizer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
text_vectorizer.set_weights(from_disk['weights'])

train_dataset = tf.data.Dataset.zip((train_feature, train_label))
train_dataset = train_dataset.map(lambda x,y: (text_vectorizer(x)[0],y),tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.zip((test_feature, test_label))
test_dataset = test_dataset.map(lambda x,y: (text_vectorizer(x)[0],y),tf.data.AUTOTUNE )
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Embedding
DIM = 8
embedding = Embedding(input_dim = MAX_TOKEN,output_dim= DIM, mask_zero=True, input_length=OUTPUT_SEQUENCE_LENGTH)

# Model:0 (Naive bayes model)
model0 = Pipeline([
    ('tfidf',TfidfVectorizer()),
    ('multino',MultinomialNB())
])

#fit and predict
model0.fit(xtrain['review'], ytrain)

# Model1
inputs  = keras.Input(shape= (100))
embedding_layer  = embedding(inputs)
x = LSTM(16, return_sequences=True)(embedding_layer)
x = LSTM(16)(x)
x = Dropout(.4)(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(1, activation = 'sigmoid')(x)

#building model
model1 = keras.Model(inputs = inputs, outputs = outputs, name = 'model1_lstm')

#compiling model
model1.compile(loss = keras.losses.binary_crossentropy,
              optimizer = keras.optimizers.Adam(),
              metrics = ['accuracy'])

EPOCHS = 10
model_dir = Path(os.getcwd()).parent /'model'
model_dir

# fit the model
history1 = model1.fit(train_dataset, epochs = EPOCHS, 
                      validation_data= test_dataset, 
                      # steps_per_epoch=int(0.1*(len(train_dataset) / EPOCHS)),
                      validation_steps=int(1*(len(test_dataset) / EPOCHS))
                    )

model1.save(model_dir/'model1.h5')
model1 = tf.keras.models.load_model(model_dir/'model1.h5')

