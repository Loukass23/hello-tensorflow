# import tensorflow as tf
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


import numpy as np

print(tf.config.list_physical_devices('GPU'))

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)

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
model.add(keras.layers.Embedding(88000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense( 16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
 
x_val = train_data[:10000]
x_train = train_data[10000:]
 
y_val = train_labels[:10000]
y_train = train_labels[10000:]

fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

results = model.evaluate(test_data, test_labels)
# print(results)

# model.save("model.h5")

def review_encode(s):
    encode = [1]

    for word in s:
        if word in words_index:
            encode.append(words_index[word.lower()])
        else:
            encode.append(2)
    return encode

# model = keras.models.load_model("model.h5")

with open("LOTR.txt") as f:
    for line in f.readlines():
        nline = line.replace(",", "").replace(".", "").replace("()", "").replace(":", "").replace("\"", "").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=words_index["<PAD>"], padding="post", maxlen=250)
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])


# testmodel
# test_review = test_data[0]
# predict = model.predict([test_review])
# print("Review: ")
# print(decode_review(test_review))
# print("Prediction:" + str(predict[0]))
# print("Actual:" + str(test_labels[0]))
# print(results)
