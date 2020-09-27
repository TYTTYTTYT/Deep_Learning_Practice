import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import timeW
    

def initiate_dataset(size=5000):
    train_set = np.empty((size, 28 * 28))
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

    return train_set / 255, train_label, train_data