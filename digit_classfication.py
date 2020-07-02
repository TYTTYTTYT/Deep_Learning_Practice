# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import time

# %%
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

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def d_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

def draw(x):
    image = np.reshape(x, (28, 28))
    plt.imshow(image, cmap='gray')

# %%
class Network_NP:

    __batch_size__ = 0

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(w @ a + b)
        return a

    def predict(self, x):
        y = self.feedforward(x)
        return np.argmax(y)

    def SGD(self, train_data, epoch, batch_size, eta, test_data=None):
        n = len(train_data)
        self.__batch_size__ = batch_size

        for i in range(epoch):
            tic = time.time()

            random.shuffle(train_data)
            for j in range(0, n, batch_size):
                batch = train_data[j: j + batch_size]
                self.update_batch(batch, eta)

            if test_data:
                self.evaluate(test_data)
            
            elapse = time.time() - tic
            print('epoch ' + str(i) + 'finished!')
            print('time usage: ' + str(elapse))
        
        return

    def update_batch(self, batch, eta):
        delta_w = [np.zeros(w.shape) for w in self.weights]
        delta_b = [np.zeros(b.shape) for b in self.biases]

        for x, y in batch:
            d_w, d_b = self.bp(x, y)
            delta_w = [origin + new for origin, new in zip(delta_w, d_w)]
            delta_b = [origin + new for origin, new in zip(delta_b, d_b)]

        self.weights = [w - eta / len(batch) * d for w, d in zip(self.weights, delta_w)]
        self.biases = [b - eta / len(batch) * d for b, d in zip(self.biases, delta_b)]
        
        return

    def bp(self, x, y):
        d_ws = [np.empty(w.shape) for w in self.weights]
        d_bs = [np.empty(b.shape) for b in self.biases]
        a = x
        activations = [x]
        zs = []

        for w, b in zip(self.weights, self.biases):
            z = w @ a
            z = z + b
            a = sigmoid(z)
            activations.append(a)
            zs.append(z)
        
        d_cost = a - y
        delta = d_cost * d_sigmoid(zs[-1])
        d_bs[-1] = delta
        d_ws[-1] = d_bs[-1] @ activations[-2].T

        for l in range(2, self.num_layers):
            delta = d_sigmoid(zs[-l]) * self.weights[-l + 1].T @ delta
            d_ws[-l] = delta @ activations[-l - 1].T
            d_bs[-l] = delta
        
        return d_ws, d_bs

    def evaluate(self, test_data):
        n = len(test_data)
        correct = 0
        wrong = 0
        for x, y in test_data:
            if self.predict(x) == np.argmax(y):
                correct += 1
            else:
                wrong += 1
        print('accuracy = ' + str(correct / (correct + wrong)))
        return


# %%
class Network_torch:

    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        self.weights = [torch.randn((n, d), dtype=torch.double)
                        for n, d in zip(self.sizes[1:], self.sizes[:-1])]
        self.biases = [torch.randn((n, 1), dtype=torch.double)
                       for n in sizes[1:]]

    def feedforward(self, x):
        activation = x
        for w, b in zip(self.weights, self.biases):
            activation = (w @ activation + b).sigmoid()
        
        return activation

    def predict(self, x):
        with torch.no_grad():
            activation = self.feedforward(x)
            values, indices = torch.max(activation, dim=0)

        return indices

    def featurelize(self, labels):
        n = len(labels)
        features = torch.zeros(self.sizes[-1], n, device=self.device, dtype=torch.double)

        for i in range(n):
            features[labels[i], i] = 1

        return features

    def evaluate(self, test_set, test_label):
        with torch.no_grad():
            test_set = torch.from_numpy(test_set.T).to(self.weights[0].device)
            test_label = torch.from_numpy(test_label).to(self.weights[0].device)

            prediction = self.predict(test_set)
            comp = prediction == test_label

            accuracy = comp.sum().double() / comp.shape[0]
            accuracy = accuracy.cpu().data.tolist()

            print('accuracy: ' + str(accuracy))

            return accuracy

    def MSE(self, activations, y):
        d = y - activations
        cost = d.square().sum() / activations.shape[1] / 2

        return cost

    def cross_entropy(self, activations, y):
        cost = -(y * activations.log() + (1 - y) * (1 - activations).log()).sum() / activations.shape[1]
        return cost

    def SGD(self, train_set, train_label, epoch, batch_size, eta,
            test_set=None, test_label=None, cost_function=None):
        if cost_function == 'MSE':
            cost_function = self.MSE
        if cost_function == 'cross_entropy':
            cost_function = self.cross_entropy
        else:
            cost_function = self.cross_entropy

        n = train_set.shape[0]
        for i in range(self.num_layers - 1):
            self.weights[i] = self.weights[i].to(self.device)
            self.weights[i].requires_grad_()
            self.biases[i] = self.biases[i].to(self.device)
            self.biases[i].requires_grad_()
        
        train_set = torch.from_numpy(train_set.T).to(self.device)
        train_label = self.featurelize(train_label).to(self.device)

        for i in range(epoch):
            tic = time.time()

            perm = torch.randperm(n)

            for j in range(0, n, batch_size):
                indices = perm[j:j + batch_size]

                activations = self.feedforward(train_set[:, indices])
                # d = train_label[:, indices] - activations
                # cost = (d.square()).sum() / batch_size / 2
                cost = cost_function(activations, train_label[:, indices])
                cost.backward()

                for k in range(self.num_layers - 1):
                    self.weights[k] = (self.weights[k] - eta * self.weights[k].grad).detach()
                    self.weights[k].requires_grad_()
                    self.biases[k] = (self.biases[k] - eta * self.biases[k].grad).detach()
                    self.biases[k].requires_grad_()
            
            if test_set is not None:
                self.evaluate(test_set, test_label)

            elapse = time.time() - tic
            print('epoch ' + str(i) + ' finished!')
            print('time usage: ' + str(elapse))

        for i in range(self.num_layers - 1):
            self.weights[i] = self.weights[i].to(torch.device('cpu')).detach()
            self.biases[i] = self.biases[i].to(torch.device('cpu')).detach()
        return


# %%
train_set, train_label, train_data = initiate_dataset(60000)


# %%
net = Network_torch([784, 100, 100, 100, 10])
net.SGD(train_set[:50000], train_label[:50000], 30, 10, 0.1, train_set[55000:56000], train_label[55000:56000])

# %%
net = Network_NP([784, 1000, 100, 30, 30, 30, 10])
net.SGD(train_data[:50000], 30, 10, 3, train_data[55000:56000])

# %%
