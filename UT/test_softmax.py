import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
from collections import OrderedDict

sys.path.append("../..")
import src_pytorch as d2l

# %% load data
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# %% define&initial module
num_inputs = 784
num_outputs = 10
# class LinearNet(nn.Module):
#     def __init__(self, num_inputs, num_outputs):
#         super(LinearNet, self).__init__()
#         self.linear = nn.Linear(num_inputs, num_outputs)
#     def forward(self, x): # x shape: (batch, 1, 28, 28)
#         y = self.linear(x.view(x.shape[0], -1))
#         return y
# net = LinearNet(num_inputs, num_outputs)
net = nn.Sequential(
    # FlattenLayer(),
    # nn.Linear(num_inputs, num_outputs)
    OrderedDict([
        ('flatten', d2l.FlattenLayer()),
        ('linear', nn.Linear(num_inputs, num_outputs))])
)
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

# %% define loss function
loss = nn.CrossEntropyLoss()

# %% define optimise function
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
num_epochs = 5
d2l.train_network(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
# %% show result
X, y = iter(test_iter).next()
true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
d2l.show_shm_fig(X[0:9], titles[0:9])