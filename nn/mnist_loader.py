import numpy as np
import random
from keras.datasets import mnist
from keras.utils import to_categorical

def load():
    
    # Load MNIST data using Keras
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Preprocess the data
    # Reshape and normalize image data
    train_images = train_images.reshape(train_images.shape[0], 28*28)
    train_images = train_images.astype('float32') / 255
    test_images = test_images.reshape(test_images.shape[0], 28*28)
    test_images = test_images.astype('float32') / 255

    # Convert labels to one-hot encoding
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    # Convert data to a list of tuples as required by the Network class
    training_data = list(zip([np.reshape(x, (784, 1)) for x in train_images], [np.reshape(y, (10, 1)) for y in train_labels]))
    test_data = list(zip([np.reshape(x, (784, 1)) for x in test_images], [np.reshape(y, (10, 1)) for y in test_labels]))
    return (training_data, test_data)
