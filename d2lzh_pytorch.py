# -*- coding: utf-8 -*-
import sys
from abc import ABC

import torch
from torch import nn
import torchvision
from IPython import display
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import torch.utils.data as data


def load_data_fashion_mnist(batch_size, resize=None, root='~/Datasets/FashionMNIST'):
    """Download the fashion mnist dataset and then load into memory."""
    if sys.platform.startswith('win'):
        num_workers = 0  # 0
    else:
        num_workers = 4
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)

    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter


def load_custom_shm_data(batch_size, root='~/Datasets/FashionMNIST'):
    """Download the fashion mnist dataset and then load into memory."""
    if sys.platform.startswith('win'):
        num_workers = 0  # 0
    else:
        num_workers = 4

    dataset = CsvDataset(root)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_iter, test_iter


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


def sgd(params, lr, batch_size):  # d2lzh_pytorch
    for param in params:
        param.data -= lr * param.grad / batch_size


def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
    plt.ion()
    fig = plt.figure()
    plt.axis([0, 150, 0.5, 1])
    plt.grid()
    for epoch in range(num_epochs):
        adjust_learning_rate(optimizer, epoch, lr)
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            # Reset the grad
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()  # Set the parameters
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        # plt.clf()
        plt.plot(epoch, test_acc, '.r')
        # plt.grid()
        # plt.plot(epoch, train_l_sum / n, 'xb')
        plt.pause(0.0001)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % (
            epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
    plt.grid()
    plt.ioff()
    plt.grid()


def adjust_learning_rate(optimizer, epoch, lr):
    """Reduce learning rate by half every 50 epoch"""
    factor = lr * 0.7 ** (epoch // 60)
    for param_group in optimizer.param_groups:
        param_group['lr'] = factor
        print('Learning rate: ', param_group['lr'])


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress',
                   'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def get_custom_shm_labels(labels):
    text_labels = ['Fail', 'Pass', 'Vol-Wall', 'Freq-Wall', 'Marginal', '']
    return [text_labels[int(i)] for i in labels]


def use_svg_display():
    display.set_matplotlib_formats('svg')


def show_fashion_mnist(images, labels):
    # use_svg_display()
    i = 0
    _, figs = plt.subplots(4, int(len(images) / 4), figsize=(12, 8))
    figs = figs.flatten()
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((11, 11)).numpy(), cmap='RdYlGn')
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


class CsvDataset(data.Dataset):

    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_count = 0
        x = np.zeros([1, 11, 11], dtype=float)
        y = np.zeros([1], dtype=float)
        self.result_dict = {'Fail': [0], 'Pass': [1], 'Vol-Wall': [2], 'Freq-Wall': [3], 'Marginal': [4]}
        self.csv_df = pd.read_csv(csv_file, iterator=True, header=None)
        # Read data in chunck
        go = True
        while go:
            try:
                tmp_y = self.result_dict[self.csv_df.get_chunk(1).values[0][0]]
                tmp_x = self.csv_df.get_chunk(11).values
                tmp_x = self.convert_SHM_data(tmp_x)
                tmp_x = tmp_x[None, :, :]

                y = np.vstack((y, tmp_y))
                x = np.vstack((x, tmp_x))
                self.data_count += 1
            except Exception as e:
                print(type(e))
                go = False
        print('The Training Data Set Count is: ', self.data_count)
        # Reshape the data
        y = y.reshape(y.shape[0])
        x = x.reshape(-1, 1, 11, 11)

        self.X_train = torch.tensor(x, dtype=torch.float)
        self.Y_train = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        # print len(self.landmarks_frame)
        # return len(self.landmarks_frame)
        return len(self.Y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.Y_train[idx]

    def convert_SHM_data(self, tmp_np):
        tmp_np[tmp_np == 'P'] = 1.
        tmp_np[tmp_np == 'p'] = 1.
        tmp_np[tmp_np == '.'] = 0.
        tmp_np[tmp_np == '*'] = 1.
        tmp_np[tmp_np == '#'] = 0.
        tmp_np = tmp_np.astype(float)
        return tmp_np


def corr2d(X, K):  #
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y


def pool2d(X, pool_size, mode='max'):
    X = X.float()
    p_h, p_w = pool_size
    Y = torch.zeros(X.shape[0] - p_h + 1, X.shape[1] - p_w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y


class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 3, padding=(1,1)),  # in_channels, out_channels, kernel_size, padding
            nn.ReLU(),
            # nn.MaxPool2d(2, 2),  # kernel_size, stride
            nn.Conv2d(6, 16, 3, padding=(1,1)),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 11 * 11, 80),
            nn.ReLU(),
            nn.Linear(80, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output
