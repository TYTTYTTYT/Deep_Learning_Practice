# %%
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch
from torch import optim
import random
import time
from utils import load_1d_data
from utils import load_2d_data
from utils import draw
from utils import table_draw

# %%
class Simple_Generator(nn.Module):
    def __init__(self):
        super(Simple_Generator, self).__init__()
        
        # define hidden linear layers
        self.fc1 = nn.Linear(100, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        
        # final fully-connected layer
        self.fc4 = nn.Linear(128, 28 * 28)
        
        # dropout layer
        self.dropout = nn.Dropout(0.3)

        self.criterion = nn.BCEWithLogitsLoss()

        return

    def forward(self, x):
        # all hidden layers
        x = F.leaky_relu(self.fc1(x), 0.2)  # (input, negative_slope=0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)
        # final layer with tanh applied
        out = F.tanh(self.fc4(x))

        return out

class Simple_Discriminator(nn.Module):
    def __init__(self):
        super(Simple_Discriminator, self).__init__()
        
        # define hidden linear layers
        self.fc1 = nn.Linear(784, 32 * 4)
        self.fc2 = nn.Linear(32 * 4, 32 * 2)
        self.fc3 = nn.Linear(32 * 2, 32)
        
        # final fully-connected layer
        self.fc4 = nn.Linear(32, 1)
        
        # dropout layer
        self.dropout = nn.Dropout(0.3)

        self.criterion = nn.BCEWithLogitsLoss()

        return
        
    def forward(self, x):
        # flatten image
        x = x.view(-1, 28 * 28)
        # all hidden layers
        x = F.leaky_relu(self.fc1(x), 0.2)  # (input, negative_slope=0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)
        # final layer
        out = self.fc4(x)

        return out

class Simple_GAN(nn.Module):
    def __init__(self):
        super(Simple_GAN, self).__init__()
        self.generator = Simple_Generator()
        self.discriminator = Simple_Discriminator()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = self.generator.forward(x)

        return x

    def noise(self, num_samples):
        return torch.rand(num_samples, 100, device=(self.device)) * 2 - 1

    def train(self, train_set, batch_size, num_epoche, g_eta, g_momentum, d_eta, d_momentum, show=False):
        train_set = train_set.reshape(-1, 28 * 28).to(self.device) * 2 - 1
        self.to(self.device)
        print("trainning on " + str(self.device))

        labels = torch.cat((torch.zeros(batch_size, 1), torch.ones(batch_size, 1)), 0)
        labels = labels.to(self.device)
        fake_labels = torch.ones(batch_size, 1).to(self.device)

        g_optimizer = optim.Adam(self.generator.parameters(), lr=g_eta)
        d_optimizer = optim.Adam(self.discriminator.parameters(), lr=d_eta)
        num_set = train_set.size()[0]

        for epoch in range(num_epoche):
            tic = time.time()

            perm = perm = torch.randperm(num_set)
            for j in range(0, num_set, batch_size):
                indices = perm[j:j + batch_size]

                # optimize generator
                g_optimizer.zero_grad()

                noise = self.noise(batch_size)

                g_out = self.generator(noise)
                d_out = self.discriminator(g_out)

                g_loss = self.generator.criterion(d_out, fake_labels)
                g_loss.backward()
                g_optimizer.step()

                # optimize discriminator
                d_optimizer.zero_grad()

                fake = g_out.detach()
                real = train_set[indices].squeeze()
                batch = torch.cat((fake, real), 0)

                d_out = self.discriminator(batch)
                d_loss = self.discriminator.criterion(d_out, labels)
                d_loss.backward()
                d_optimizer.step()

            elapse = time.time() - tic
            print('epoch ' + str(epoch) + ' finished!')
            print('time usage: ' + str(elapse))

            if show is True:
                with torch.no_grad():
                    x = self.forward(self.noise(9))
                x = x.to(torch.device('cpu'))
                table_draw(x, 3, 3)
        
        self.to(torch.device('cpu'))


# %%
class CNN_Generator(nn.Module):
    def __init__(self):
        super(CNN_Generator, self).__init__()
        # The decoder
        self.t_conv1 = nn.ConvTranspose2d(1, 16, 2, stride=2, bias=False)
        # self.bn1 = nn.BatchNorm2d(16)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2, bias=False)
        # self.bn2 = nn.BatchNorm2d(1)

        return

    def forward(self, x):
        x = F.leaky_relu(self.t_conv1(x))
        x = F.sigmoid(self.t_conv2(x))

        return x

class CNN_Discriminator(nn.Module):
    def __init__(self):
        super(CNN_Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, stride=(3, 3))
        # self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 16, 3, stride=(3, 3))
        # self.bn2 = nn.BatchNorm2d(16)

        # self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(16 * 3 * 3, 120)
        self.fc2 = nn.Linear(120, 32)
        self.fc3 = nn.Linear(32, 1)

        # self.sm = nn.Softmax()

        # self.criterion = nn.NLLLoss()
        self.criterion = nn.BCELoss()

        return

    def forward(self, x):
        # Let the net to learn pool function
        # x = self.pool(F.leaky_relu(self.conv1(x)))

        # x = self.pool(F.leaky_relu(self.conv2(x)))

        x = x.reshape(-1, 1, 28, 28)

        x = F.leaky_relu(self.conv1(x))
        # x = F.leaky_relu(x)
        x = F.leaky_relu(self.conv2(x))
        # x = F.leaky_relu(x)

        x = x.view(-1, self.num_flat_features(x))   # -1 represents the batch dimension
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class CNN_GAN(nn.Module):
    def __init__(self):
        super(CNN_GAN, self).__init__()
        self.generator = CNN_Generator()
        self.discriminator = CNN_Discriminator()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = self.generator.forward(x)

        return x

    def noise(self, num_samples):
        return torch.rand(num_samples, 1, 7, 7, device=(self.device))

    def g_loss(self, d_out):
        loss = torch.mean(torch.log(1 - d_out))

        return loss

    def train(self, train_set, batch_size, num_epoche, g_eta, g_momentum, d_eta, d_momentum, show=False):
        train_set = train_set.to(self.device)
        self.to(self.device)
        print("trainning on " + str(self.device))

        labels = torch.cat((torch.zeros(batch_size, 1), torch.ones(batch_size, 1)), 0)
        labels = labels.to(self.device)
        fake_labels = torch.ones(batch_size, 1).to(self.device)

        g_optimizer = optim.Adam(self.generator.parameters(), lr=g_eta)
        d_optimizer = optim.Adam(self.discriminator.parameters(), lr=d_eta)
        num_set = train_set.size()[0]

        for epoch in range(num_epoche):
            tic = time.time()

            perm = perm = torch.randperm(num_set)
            for j in range(0, num_set, batch_size):
                indices = perm[j:j + batch_size]

                # optimize generator
                g_optimizer.zero_grad()

                noise = self.noise(batch_size)

                g_out = self.generator(noise)
                d_out = self.discriminator(g_out)

                g_loss = self.g_loss(d_out)
                g_loss.backward()
                g_optimizer.step()

                # optimize discriminator
                d_optimizer.zero_grad()

                fake = g_out.detach()
                real = train_set[indices]
                batch = torch.cat((fake, real), 0)

                d_out = self.discriminator(batch)
                d_loss = self.discriminator.criterion(d_out, labels)
                d_loss.backward()
                d_optimizer.step()

            elapse = time.time() - tic
            print('epoch ' + str(epoch) + ' finished!')
            print('time usage: ' + str(elapse))

            if show is True and epoch % 10 == 0:
                with torch.no_grad():
                    x = self.forward(self.noise(9))
                x = x.to(torch.device('cpu'))
                table_draw(x, 3, 3)
        
        self.to(torch.device('cpu'))

# %%
train_set, train_label, valid_set, valid_label, test_set, test_label = load_2d_data(50000, 0, 0)

# %%
net = Simple_GAN()

# %%
net.train(train_set, 100, 200, 0.002, 0.9, 0.002, 0.1, True)

# %%
net2 = CNN_GAN()

# %%
net2.train(train_set, 100, 20, 0.002, 0.9, 0.002, 0.1, True)

# %%
noise = net.noise(9).to(torch.device('cpu'))
with torch.no_grad():
    g_out = net.generator(noise)

# %%
print(g_out.size())

# %%
print(g_out)
# %%
table_draw(g_out, 3, 3)
# %%
net.discriminator(g_out)
# %%
net.discriminator(train_set[:10] * 2 - 1)
# %%
net.to(torch.device('cpu'))
# %%
net.discriminator.criterion(net.discriminator(g_out), torch.rand(20, 1))
# %%
