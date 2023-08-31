
import re
from pathlib import Path

import polars as pl
from sklearn.model_selection import train_test_split


def preprocess_data(data_dir:Path):
    # Read the CSV file using Polars
    df = pl.read_csv(data_dir / 'train.csv', new_columns=['polarity', 'title', 'text'])

    assert df['polarity'].max()==2
    assert df['polarity'].min()==1

    # Drop rows with null values
    df.drop_nulls()

    # Map polarity to binary values (0 for negative, 1 for positive)
    df = df.with_columns([
        pl.col('polarity').apply(lambda x: 0 if x == 1 else 1)
    ])

    # Cast polarity column to Int16
    df = df.with_columns([
        pl.col('polarity').cast(pl.Int16, strict=False)
    ])

    # Combine title and text columns to create the review column
    df = df.with_columns([
        (pl.col('title') + ' ' + pl.col('text')).alias('review')
    ])
    
    df = df.with_columns([
	(pl.col('review').str().lower())
	])
	
    # Select relevant columns
    df = df.select(['review', 'polarity'])

    # Perform text cleaning using a function
    df = df.with_columns([
        pl.col('review').apply(clean_text)
    ])

    df.write_csv(data_dir/'preprocessed_df.csv')
    
    

import re

import contractions

# Compile the regular expressions outside the function for better performance
PUNCTUATION_REGEX = re.compile(r'[^\w\s]')
DIGIT_REGEX = re.compile(r'\d')

def clean_text(x: str) -> str:
    expanded_text = contractions.fix(x)  # Expand contractions
    cleaned_text = PUNCTUATION_REGEX.sub(' ', expanded_text.lower())  # Remove punctuation after lowering
    cleaned_text = DIGIT_REGEX.sub('', cleaned_text)  # Remove digits
    return cleaned_text