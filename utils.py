import matplotlib.pyplot as plt
import numpy as np
import torch
import random

# Global variables used to store raw datas
data_set = None
data_label = None
data = None     # data is set + label

def draw(x):
    image = np.reshape(x, (28, 28))
    plt.imshow(image, cmap='gray')

    return

def table_draw(x, rows, columns):
    fig = plt.figure()
    plt.axis("off")
    index = 0
    for r in range(rows):
        for c in range(columns):
            image = np.reshape(x[index], (28, 28))
            index += 1
            fig.add_subplot(rows, columns, index)
            plt.imshow(image, cmap='gray')

    return

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

def feature(y):
    fea = np.zeros((10, 1), dtype=int)
    fea[y, 0] = 1
    return fea

def split_dataset(data_set, data_label, num_valid=500, num_test=1000):
    num_train = data_set.shape[0] - num_valid - num_test

    train_set = data_set[:num_train]
    train_label = data_label[:num_train]

    valid_set = data_set[num_train:num_train + num_valid]
    valid_label = data_label[num_train:num_train + num_valid]

    test_set = data_set[-num_test:]
    test_label = data_label[-num_test:]

    return train_set, train_label, valid_set, valid_label, test_set, test_label

def dataset_to_2d(data_set, data_label, valid_size=500, test_size=1000):
    num_train = data_set.shape[0] - valid_size - test_size

    set_2d = torch.from_numpy(data_set).type(torch.float)
    set_2d = set_2d.view(-1, 1, 28, 28)
    label_2d = torch.from_numpy(data_label)

    train_set_2d = set_2d[:num_train]
    train_label_2d = label_2d[:num_train]

    valid_set_2d = set_2d[num_train:num_train + valid_size]
    valid_label_2d = label_2d[num_train:num_train + valid_size]

    test_set_2d = set_2d[-test_size:]
    test_label_2d = label_2d[-test_size:]

    return train_set_2d, train_label_2d, valid_set_2d, valid_label_2d, test_set_2d, test_label_2d

def load_1d_data(num_train, num_valid, num_test):
    global data_set, data_label, data

    if data_set is None or data_label is None:
        data_set, data_label, data = initiate_dataset(num_test + num_train + num_valid)

    train_data = data[:num_train]
    valid_data = data[num_train:num_train + num_valid]

    return split_dataset(data_set, data_label, num_valid, num_test), train_data, valid_data

def load_2d_data(num_train, num_valid, num_test):
    global data_set, data_label, data

    if data_set is None or data_label is None:
        data_set, data_label, data = initiate_dataset(num_test + num_train + num_valid)

    return dataset_to_2d(data_set, data_label, num_valid, num_test)
