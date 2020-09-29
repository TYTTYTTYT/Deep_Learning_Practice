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
class CNNAutoencoder(nn.Module):
    def __init__(self):
        super(CNNAutoencoder, self).__init__()
       
        # Encoder
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 4, 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
       
        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)

        self.criterion = nn.MSELoss()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        return

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.t_conv1(x))
        x = F.sigmoid(self.t_conv2(x))

        return x

    def SGD(self, train_set, batch_size, num_epoche, eta, momentum, show=False):
        train_set = train_set.to(self.device)
        self.to(self.device)

        optimizer = optim.SGD(self.parameters(), lr=eta, momentum=momentum)
        num_set = train_set.size()[0]
        
        for epoch in range(num_epoche):
            tic = time.time()

            perm = perm = torch.randperm(num_set)
            for j in range(0, num_set, batch_size):
                indices = perm[j:j + batch_size]

                optimizer.zero_grad()
                out = self.forward(train_set[indices])
                loss = self.criterion(out, train_set[indices])
                loss.backward()
                optimizer.step()

            elapse = time.time() - tic

            if show is True:
                x = self.forward(train_set[:4]).detach()
                x = x.to(torch.device('cpu'))
                table_draw(x, 2, 2)

            print('epoch ' + str(epoch) + ' finished!')
            print('time usage: ' + str(elapse))

        self.to(torch.device('cpu'))

# %%
train_set, train_label, valid_set, valid_label, test_set, test_label = load_2d_data(40000, 0, 0)

# %%
net = CNNAutoencoder()
net.SGD(train_set, 10, 30, 0.1, 0.1, True)
# %%
x = net(train_set[:25]).detach()
table_draw(x, 5, 5)
# %%
table_draw(train_set[:25], 5, 5)
# %%
