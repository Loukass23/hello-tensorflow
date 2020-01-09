import tensorflow as tf
from tensorflow import keras 
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt', 'Pants', 'Pullover', 'Dress', 'Coat',
 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
# plt.imshow(train_images[3], cmap=plt.cm.binary)
plt.imshow(train_images[3])
# plt.show()
 
train_images = train_images / 255.0
test_images = test_images / 255.0

# print(train_images[3])
# print(train_labels)


# defining NN model's layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10,activation="softmax")
])
# 1 - flatening initial data input
# 2 - activation relu = rectify linear unit
# 3 - softmax bring all values to 0-1

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# train model
model.fit(train_images, train_labels, epochs=5 )
# epochs how many time the model is gonna see the information => randomly pick img to feed the NN => increase accuracy

test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Tested acc:", test_acc)


# generating a group of predictions
prediction = model.predict(test_images)


for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual" + class_names[test_labels[i]] )
    plt.title("Prediction" + class_names[np.argmax(prediction[i])])
    plt.show()

# print(class_names[np.argmax(prediction[0])])