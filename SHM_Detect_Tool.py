# -*- coding: utf-8 -*-
###################################################
# SHM Detect Tool                                 #
# Version: Beta 0.1                               #
#                                                 #
# Sep. 02, 2020                                   #
# A Tool to Detect the Result of SHM              #
###################################################
import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys

sys.path.append("..")
import d2lzh_pytorch as d2l

__version__ = 'SHM Detect Tool Beta V0.1'
__author__ = 'zhouchao486@gmail.com'

# %% load data
batch_size = 24  # 256
train_iter, test_iter = d2l.load_custom_shm_data(batch_size)  # d2l.load_data_fashion_mnist(batch_size)

# %% define&initial module
num_inputs, num_outputs, num_hiddens = 121, 2, 64  # 784, 10, 256
net = nn.Sequential(
    d2l.FlattenLayer(),
    nn.Linear(num_inputs, num_hiddens),
    nn.ReLU(),
    nn.Linear(num_hiddens, num_outputs),
)
for params in net.parameters():
    init.normal_(params, mean=0, std=0.01)

# %% define loss function
loss = torch.nn.CrossEntropyLoss()

# %% optimise function
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)

# %% run training
num_epochs = 60
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

# %% show result
X, y = iter(test_iter).next()
true_labels = d2l.get_custom_shm_labels(y.numpy())  # d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_custom_shm_labels(
    net(X).argmax(dim=1).numpy())  # d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
d2l.show_fashion_mnist(X[0:9], titles[0:9])
