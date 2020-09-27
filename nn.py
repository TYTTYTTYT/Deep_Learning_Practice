# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import time

# %%
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 3, stride=(1, 1))
        self.conv2 = nn.Conv2d(6, 16, 3, stride=(1, 1))

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.sm = nn.Softmax()

        self.criterion = nn.NLLLoss()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, self.num_flat_features(x))   # -1 represents the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.sm(self.fc3(x))
        return x

    def predict(self, x):
        with torch.no_grad():
            activation = self.forward(x)
            values, indices = torch.max(activation, dim=1)

        return indices

    def evaluate(self, valid_set, valid_label):
        with torch.no_grad():
            prediction = self.predict(valid_set)
            comp = prediction == valid_label

            accuracy = comp.sum().double() / comp.shape[0]
            accuracy = accuracy.cpu().data.tolist()

            print('CNN\'s accuracy: ' + str(accuracy))

            return accuracy

    def SGD(self, train_set, train_label, batch_size, num_epoche, eta,
            valid_set=None, valid_label=None):
        optimizer = optim.SGD(net.parameters(), lr=eta)
        num_set = train_set.size()[0]
        
        for epoch in range(num_epoche):
            tic = time.time()

            perm = perm = torch.randperm(num_set)
            for j in range(0, num_set, batch_size):
                indices = perm[j:j + batch_size]

                optimizer.zero_grad()
                out = net(train_set[indices])
                loss = net.criterion(out, train_label[indices])
                loss.backward()
                optimizer.step()

            elapse = time.time() - tic
            print('epoch ' + str(epoch) + ' finished!')
            print('time usage: ' + str(elapse))
            
            if valid_set is not None and valid_label is not None:
                self.evaluate(valid_set, valid_label)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# %%
net = Net()
print(net)

# %%
data = torch.randn(3, 1, 28, 28)
print(data)
out = net(data)
print(out)
print(net.predict(data))

# %%
params = list(net.parameters())
print(len(params))
print(params[0].size())
print(out.size())

# %%
fakes = torch.randn(3, 10) / 2 + 1
target = torch.tensor([1, 2, 3])
loss = net.criterion(out, target)
print(loss)

# %%
loss.backward()

# %%
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

# %%
optimizer = optim.SGD(net.parameters(), lr=0.01)

optimizer.zero_grad()
out = net(data)
loss = net.criterion(out, target)
loss.backward()
optimizer.step()
# %%
fake_data = torch.randn(10, 1, 28, 28)
fake_label = torch.tensor([1, 2, 3, 4, 5, 6, 7, 4, 5, 0])
net = Net()

net.SGD(fake_data, fake_label, 2, 100, 0.01)
# %%
