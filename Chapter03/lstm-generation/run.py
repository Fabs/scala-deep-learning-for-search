# keras module for building LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku
from keras import utils as np_utils
from tensorflow.keras.utils import to_categorical
import keras

from numpy.random import seed

import pandas as pd
import numpy as np
import string, os
import sys

import warnings

def clean_text(txt):
    txt = "".join(v for v in txt if v not in string.punctuation).lower()
    txt = txt.encode("utf8").decode("ascii",'ignore')
    return txt

def get_sequence_of_tokens(tokenizer, corpus):
    ## tokenization
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    ## convert data to sequence of tokens
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences, total_words

def generate_padded_sequences(input_sequences, total_words):
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = to_categorical(label, num_classes=total_words)
    return predictors, label, max_sequence_len

def generate_text(tokenizer, seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predict_x = model.predict(token_list)
        classes_x = np.argmax(predict_x,axis=1)
        predicted = classes_x

        output_word = ""
        for word,index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " "+output_word
    return seed_text.title()

def train():
    warnings.filterwarnings("ignore")
    warnings.simplefilter(action='ignore', category=FutureWarning)

    curr_dir = './data/'
    all_queries = []
    for filename in os.listdir(curr_dir):
        if 'queries' in filename:
            with open(curr_dir + '/' + filename) as fp:
                lines = fp.readlines()
                all_queries.extend(lines)

    print(len(all_queries))


    corpus = [clean_text(x) for x in all_queries][1:500]
    print(corpus[:10])

    tokenizer = Tokenizer()

    inp_sequences, total_words = get_sequence_of_tokens(tokenizer, corpus)
    print(inp_sequences[:10])

    predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences, total_words)

    def create_model(max_sequence_len, total_words):
        input_len = max_sequence_len - 1
        model = Sequential()

        # Add Input Embedding Layer
        model.add(Embedding(total_words, 10, input_length=input_len))

        # Add Hidden Layer 1 - LSTM Layer
        model.add(LSTM(100))
        model.add(Dropout(0.1))

        # Add Output Layer
        model.add(Dense(total_words, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam')

        return model

    model = create_model(max_sequence_len, total_words)
    print(model.summary())

    model.fit(predictors, label, epochs=100, verbose=5)
    model.save('model.h5')

def predict(word, size):
    warnings.filterwarnings("ignore")
    warnings.simplefilter(action='ignore', category=FutureWarning)

    curr_dir = './data/'
    all_queries = []
    for filename in os.listdir(curr_dir):
        if 'queries' in filename:
            with open(curr_dir + '/' + filename) as fp:
                lines = fp.readlines()
                all_queries.extend(lines)

    print(len(all_queries))

    corpus = [clean_text(x) for x in all_queries][1:500]

    tokenizer = Tokenizer()

    inp_sequences, total_words = get_sequence_of_tokens(tokenizer, corpus)

    predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences, total_words)

    model = keras.models.load_model("model.h5")

    print(generate_text(tokenizer, word, size, model, max_sequence_len))

# From https://www.kaggle.com/code/shivamb/beginners-guide-to-text-generation-using-lstms/notebook
if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "train":
        train()
    else:
        word = sys.argv[2]
        size = int(sys.argv[3])
        predict(word, size)


