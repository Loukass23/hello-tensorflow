import tensorflow as tf
from tensorflow import keras 
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels), (train_images, train_labels) = data.load_data()

class_names = ['T-shirt', 'Pants', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
plt.imshow(train_images[7])
plt.show()
print(train_labels)