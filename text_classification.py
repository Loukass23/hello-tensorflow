import tensorflow as tf
from tensorflow import keras 
import numpy as np


data = keras.datasets.imdb

(train_data, train_laels), (test_data, test_labels) = data.load_data(num_words=10000)

# print(train_data)

words_index = data.get_word_index()
words_index = {k: (v + 3) for k, v in words_index.items()}
words_index["<PAD>"] = 0
words_index["<START>"] = 1
words_index["<UNK>"] = 2
words_index["<UNUSED>"] = 3

# word mapping
reverse_word_index = dict([value, key] for (key, value) in words_index.items())


#disparity in size so NN model won't work
# print(len(test_data[0]), len(test_data[1]))

# only keep review of fixed size
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=words_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=words_index["<PAD>"], padding="post", maxlen=250)
# print(len(train_data), len(test_data))

def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

# print(decode_review(test_data[0]))

# define Model and layers
model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense( 16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))
