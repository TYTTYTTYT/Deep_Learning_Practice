# %%
import matplotlib.pyplot as plt
import numpy as np
import torch

# %%
def initiate_dataset(size=5000):
    train_set = np.empty((size, 28 * 28), dtype=int)
    train_label = np.empty(size, dtype=int)
    with open('origin.data', 'rb') as f:
        f.seek(16)
        raw = f.read(size * 28 * 28)

    index = 0
    for i in range(size):
        for j in range(28 * 28):
            train_set[i, j] = raw[index]
            index += 1

    with open('train.label', 'rb') as f:
        f.seek(8)
        raw = f.read(size)

    for i in range(size):
        train_label[i] = raw[i]

    train_data = [[np.reshape(x, (784, 1)) / 255, feature(y)] for x, y in zip(train_set, train_label)]

    return train_set, train_label, train_data

def feature(y):
    fea = np.zeros((10, 1), dtype=int)
    fea[y, 0] = 1
    return fea

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def d_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

def draw(x):
    image = np.reshape(x, (28, 28))
    plt.imshow(image, cmap='gray')

# %%
train_set, train_label, train_data = initiate_dataset(100)


# %%
class Network:

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(w @ a + b)
        return a


# %%
net = Network([784, 16, 16, 10])

# %%
