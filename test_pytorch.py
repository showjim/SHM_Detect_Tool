import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
import torch.utils.data as Data
import torch.nn as nn
from torch.nn import init
import torch.optim as optim

# %% generate data
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
# %% read data
batch_size = 10
dataset = Data.TensorDataset(features, labels)
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
# for X, y in data_iter:
#     print(X, y)
#     break
# %% defeine module
net = nn.Sequential(nn.Linear(num_inputs, 1))
print(net)
print(net[0])
# %% initial module
init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0)  # net[0].bias.data.fill_(0)
# %% define loss function
loss = nn.MSELoss()
# %% define optmise function
optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)
# %% training modle
num_epochs = 6
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad()  # equal to net.zero_grad()
        l.backward()
        optimizer.step()
        # # adjust lr
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] *= 0.1
    print('epoch %d, loss: %f' % (epoch, l.item()))
dense = net[0]
print(true_w, dense.weight)
print(true_b, dense.bias)
