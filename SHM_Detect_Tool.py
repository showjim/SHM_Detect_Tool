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
batch_size = 80  # 256
filename = r'custom_SHM_data.csv'
train_iter, test_iter = d2l.load_custom_shm_data(batch_size, filename)  # d2l.load_data_fashion_mnist(batch_size)

# %% define&initial module
# num_inputs, num_outputs, num_hiddens = 121, 2, 64  # 784, 10, 256
# net = nn.Sequential(
#     d2l.FlattenLayer(),
#     nn.Linear(num_inputs, num_hiddens),
#     nn.ReLU(),
#     nn.Linear(num_hiddens, num_outputs),
# )
# for params in net.parameters():
#     init.normal_(params, mean=0, std=0.01)

net = d2l.LeNet()

# %% define loss function
loss = torch.nn.CrossEntropyLoss()

# %% optimise function
lr = 0.001
# optimizer = torch.optim.SGD(net.parameters(), lr=lr)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# %% run training
num_epochs = 80 #320
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, lr, optimizer)

# %% show result
X, y = iter(test_iter).next()
true_labels = d2l.get_custom_shm_labels(y.numpy())  # d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_custom_shm_labels(
    net(X).argmax(dim=1).numpy())  # d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
d2l.show_fashion_mnist(X[0:36], titles[0:36])
