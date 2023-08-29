# Amazon review sentiment analysis
![GitHub Stars](https://img.shields.io/github/stars/tikendraw/amazon-review-sentiment-analysis)

Welcome to the Amazon Review Sentiment Analysis project! This repository contains code for training a sentiment analysis model on a large dataset of Amazon reviews using Long Short-Term Memory (LSTM) neural networks. The trained model can predict the sentiment (positive or negative) of Amazon reviews. The dataset used for training consists of over 2 million reviews, totaling 2.6 GB of data.


## Table of Contents
* Introduction
* Dataset
* Model
* Getting Started
    * Prerequisites
    * Training
    * Prediction
    * Running the Streamlit App
* Contributing
* Acknowledgements

## Introduction
Sentiment analysis is the process of determining the sentiment or emotion expressed in a piece of text. In this project, we focus on predicting whether Amazon reviews are positive or negative based on their text content. We use LSTM neural networks, a type of recurrent neural network (RNN), to capture the sequential patterns in the text data and make accurate sentiment predictions.

## Dataset
The dataset used for this project is a massive collection of Amazon reviews, comprising more than 2 million reviews with a total size of 2.6 GB. The dataset is [ here](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews). It contains both positive and negative reviews, making it suitable for training a sentiment analysis model.

### Challenges 
* Dataset is very larget (2.6 GB) with 2.6 Million Reviews
* Machine's resources are limiting as loading multiple variables with processed data is eating up RAM

### Work Arounds
* Used polars for data manipulation and Preprocessings ( Uses Parallel computation, Doesn't load data on memory)

## Model
The sentiment analysis model is built using TensorFlow and Keras libraries. We employ LSTM layers to effectively capture the sequential nature of text data. The model is trained on the labeled Amazon reviews dataset, and its performance is evaluated using various metrics such as accuracy, precision, recall, and F1-score.

## Model architectures
```
Model: "model_lstm"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_3 (InputLayer)        [(None, 175)]             0         
                                                                 
 embedding_layer (Embedding)  (None, 175, 8)           2400000   
                                                                 
 lstm_layer_1 (LSTM)         (None, 175, 16)           1600      
                                                                 
 lstm_layer_2 (LSTM)         (None, 16)                2112      
                                                                 
 dropout_layer (Dropout)     (None, 16)                0         
                                                                 
 dense_layer_1 (Dense)       (None, 64)                1088      
                                                                 
 dense_layer_2_final (Dense)  (None, 1)                65        
                                                                 
=================================================================
Total params: 2,404,865
Trainable params: 2,404,865
Non-trainable params: 0
_________________________________________________________________
```
## Model Performance


## Getting Started
Follow these steps to get started with the project:

### Prerequisites
* Python 3.x
* TensorFlow
* Keras
* Polars
* Streamlit

You can install the required dependencies using the following command:

```
pip install -r requirements.txt
```

### Training
To train the LSTM model, run the train.py script:

```
python3 train.py
```
This script will preprocess the dataset, train the model, and save the trained weights to disk.

### Prediction

To use the trained model for making predictions on new reviews, run the predict.py script:

```
python3 predict.py
```
### Running the Streamlit App
We've also provided a user-friendly Streamlit app to interact with the trained
Model. Run the app using the following command:
```
streamlit run app.py
```
This will launch a local web app where you can input your own Amazon review and see the model's sentiment prediction.

## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or create a pull request.


## Acknowledgements
We would like to express our gratitude to the open-source community for providing invaluable resources and tools that made this project possible.

Don't Forget to Star!
If you find this project interesting or useful, please consider starring the repository. Your support is greatly appreciated!

Star

Happy coding!

Your Name
Your Contact Info
Date