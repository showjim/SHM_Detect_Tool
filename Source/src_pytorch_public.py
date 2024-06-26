# -*- coding: utf-8 -*-
import sys
import math
import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import torch.utils.data as data
import torch.nn.functional as F

# yes, I had an M1 device~~~
device = "mps" if torch.backends.mps.is_available() else "cpu"


def load_custom_shm_data(batch_size, root='~/Datasets/FashionMNIST'):
    """Download the fashion mnist dataset and then load into memory."""
    if sys.platform.startswith('win'):
        num_workers = 0  # 0
    else:
        num_workers = 0

    dataset = CsvDataset(root)
    train_dataset, test_dataset = data.random_split(dataset, (dataset.__len__() - 100, 100))
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                             num_workers=num_workers, drop_last=True) # add drop_last to prevent error when BN size == 1, ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 42])
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_iter, test_iter


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


def evaluate_accuracy(data_iter, net):
    net.eval()
    # net.train()
    acc_sum, n, acc_sum_pf = 0.0, 0, 0.0
    for X, y in data_iter:
        # X = X.to(device)
        # y = y.to(device)
        # acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        y_hat = net(X)
        y_hat = reformat_output(y_hat)
        # y_hat[y_hat >= 0.5] = 1
        # y_hat[y_hat < 0.5] = 0
        check_result = torch.sum(y_hat == y, 1)
        check_result_pf = torch.logical_and(y_hat[:, 0] == y[:, 0], y_hat[:, 1] == y[:, 1])
        acc_sum += (check_result == 6).float().sum().item()
        acc_sum_pf += check_result_pf.float().sum().item()
        n += y.shape[0]
    net.train()
    return acc_sum / n, acc_sum_pf / n


def sgd(params, lr, batch_size):  # d2lzh_pytorch
    for param in params:
        param.data -= lr * param.grad / batch_size


def train_network(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
    plt.ion()
    fig = plt.figure()
    plt.axis([0, 100, 0., 1.])
    plt.grid()
    for epoch in range(num_epochs):
        # adjust_learning_rate(optimizer, epoch, lr)
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            # X = X.to(device)
            # y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)  # .sum()
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
            # train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()

            y_hat = reformat_output(y_hat)

            check_result = torch.sum(y_hat == y, 1)
            train_acc_sum += (check_result == 6).sum().item()
            n += y.shape[0]
        test_acc, test_acc_pf = evaluate_accuracy(test_iter, net)
        optimizer.zero_grad()
        # plt.clf()
        plt.plot(epoch, test_acc, '.r')
        plt.plot(epoch, test_acc_pf, '+b')
        # plt.grid()
        # plt.plot(epoch, train_l_sum / n, 'xb')
        plt.pause(0.001)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, test acc P/F %.3f' % (
            epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc, test_acc_pf))
    plt.grid()
    plt.ioff()
    plt.grid()


def reformat_output(y_hat):
    a, _ = torch.max(y_hat[:, 0:2], 1)
    # process Pass/Fail label
    y_hat[:, 0] = y_hat[:, 0] / a
    y_hat[:, 1] = y_hat[:, 1] / a
    y_hat[y_hat[:, 0] < 1., 0] = 0
    y_hat[y_hat[:, 1] < 1., 1] = 0
    # y_hat[y_hat[:, 0] >= a, 0] = 1
    # y_hat[y_hat[:, 0] < a, 0] = 0
    # y_hat[y_hat[:, 1] >= a, 1] = 1
    # y_hat[y_hat[:, 1] < a, 1] = 0

    y_hat[y_hat >= 0.5] = 1
    y_hat[y_hat < 0.5] = 0
    return y_hat


def get_custom_shm_labels(labels, type):
    text_labels = ['Fail', 'Pass', 'Vol', 'Freq', 'Marginal', 'Hole']
    result_list = []
    for i in range(len(labels)):
        result = type + ': '
        for j in range(len(labels[i])):
            if labels[i][j] == 1.:
                result = result + '-' + text_labels[j]
        result_list.append(result)
    return result_list
    # return [text_labels[int(i)] for i in labels]


def show_shm_fig(images, labels):
    # use_svg_display()
    _, figs = plt.subplots(5, int(len(images) / 5), figsize=(12, 8))
    plt.tight_layout()
    figs = figs.flatten()
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((img.shape[1], img.shape[2])).numpy(), cmap='RdYlGn')
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
        y = np.zeros([6], dtype=float)
        y[0] = 1
        self.result_dict = {'Fail': 0, 'Pass': 1, 'Vol': 2, 'Freq': 3, 'Marginal': 4, 'Hole': 5}
        # self.result_dict = {'Fail':                   [1, 0, 0, 0, 0, 0],
        #                     'Pass':                   [0, 1, 0, 0, 0, 0],
        #                     'Fail-Vol':               [1, 0, 1, 0, 0, 0],
        #                     'Fail-Freq':              [1, 0, 0, 1, 0, 0],
        #                     'Fail-Marginal':          [1, 0, 0, 0, 1, 0],
        #                     'Fail-Hole':              [1, 0, 0, 0, 0, 1],
        #                     'Pass-Marginal':          [0, 1, 0, 0, 1, 0],
        #                     'Fail-Vol-Freq':          [1, 0, 1, 1, 0, 0],
        #                     'Fail-Vol-Hole':          [1, 0, 1, 0, 0, 1],
        #                     'Fail-Vol-Marginal':      [1, 0, 1, 0, 1, 0],
        #                     'Fail-Freq-Marginal':     [1, 0, 0, 1, 1, 0],
        #                     'Fail-Freq-Hole':         [1, 0, 0, 1, 0, 1],
        #                     'Fail-Marginal-Hole':     [1, 0, 0, 0, 1, 1]}
        self.csv_df = pd.read_csv(csv_file, iterator=True, header=None)
        # Read data in chunck
        go = True
        while go:
            try:
                cur_result = self.csv_df.get_chunk(1).values[0][0]
                tmp_y = self.convert_result(cur_result)
                tmp_x = self.csv_df.get_chunk(11).values
                tmp_x = self.convert_SHM_data(tmp_x)
                tmp_x = tmp_x[None, :, :]

                y = np.vstack((y, tmp_y))
                x = np.vstack((x, tmp_x))
                self.data_count += 1
                if self.data_count == 599:
                    print("OK")
            except Exception as e:
                print(type(e))
                go = False
        print('The Training Data Set Count is: ', self.data_count)
        # Reshape the data
        y = y.reshape(y.shape[0], y.shape[1])
        x = x.reshape(-1, 1, 11, 11)

        self.X_train = torch.tensor(x, dtype=torch.float)
        self.Y_train = torch.tensor(y, dtype=torch.float)

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

    def convert_result(self, text:str):
        result = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        text_list = text.split('-')
        dict_keys = self.result_dict.keys()
        for i in range(len(text_list)):
            if text_list[i] in dict_keys:
                result[self.result_dict[text_list[i]]] = 1.0
            else:
                print('Wrong Key Word Found in Training Data')
        return result


class CsvDataset_Test(data.Dataset):

    def __init__(self, csv_file, mode='training'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_count = 0
        x = np.zeros([1, 11, 11], dtype=float)
        # y = [['Dummy']] #np.zeros([1], dtype=str)
        y = []
        self.raw_dict = {}
        self.result_dict = {'Fail': 0, 'Pass': 1, 'Vol': 2, 'Freq': 3, 'Marginal': 4, 'Hole': 5}

        self.csv_df = pd.read_csv(csv_file, iterator=True, header=None)
        # Read data in chunck
        go = True
        while go:
            try:
                tmp_y = [self.csv_df.get_chunk(1).values[0][0]]
                tmp_x = self.csv_df.get_chunk(11).values
                self.raw_dict[tmp_y[0]] = tmp_x.tolist()
                tmp_x = self.convert_SHM_data(tmp_x)
                tmp_x = tmp_x[None, :, :]

                y.append(tmp_y)
                if self.data_count == 0:
                    x = tmp_x
                else:
                    x = np.vstack((x, tmp_x))
                self.data_count += 1
            except Exception as e:
                print(type(e))
                go = False
        print('The Training Data Set Count is: ', self.data_count)
        # Reshape the data
        # y = y.reshape(y.shape[0], y.shape[1])
        # x = x.reshape(-1, 1, 11, 11)
        x = x.reshape(-1, 1, np.size(x, 1), np.size(x, 2))

        self.X_train = torch.tensor(x, dtype=torch.float)
        self.Y_train = y  # np.array(y) #torch.tensor(y)
        pass

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


class CsvDataset_Test_Serial(data.Dataset):

    def __init__(self, shmoo_body, shmoo_title):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.data_count = 0
        # x = np.zeros([1, 11, 11], dtype=float)
        # y = [['Dummy']] #np.zeros([1], dtype=str)
        # y = []
        self.raw_dict = {}
        self.result_dict = {'Fail': 0, 'Pass': 1, 'Vol': 2, 'Freq': 3, 'Marginal': 4, 'Hole': 5}

        # self.csv_df = pd.read_csv(csv_file, iterator=True, header=None)
        # # Read data in chunck
        # go = True
        # while go:
        #     try:
        #         tmp_y = [self.csv_df.get_chunk(1).values[0][0]]
        #         tmp_x = self.csv_df.get_chunk(11).values
        #         self.raw_dict[tmp_y[0]] = tmp_x.tolist()
        #         tmp_x = self.convert_SHM_data(tmp_x)
        #         tmp_x = tmp_x[None, :, :]
        #
        #         y.append(tmp_y)
        #         if self.data_count == 0:
        #             x = tmp_x
        #         else:
        #             x = np.vstack((x, tmp_x))
        #         self.data_count += 1
        #     except Exception as e:
        #         print(type(e))
        #         go = False
        # print('The Training Data Set Count is: ', self.data_count)
        # Reshape the data
        # y = y.reshape(y.shape[0], y.shape[1])
        # x = x.reshape(-1, 1, 11, 11)
        self.raw_dict[shmoo_title] = shmoo_body

        tmp_x = np.asarray(shmoo_body)
        x = self.convert_SHM_data(tmp_x)
        x = x.reshape(-1, np.size(x, 0), np.size(x, 1))
        x = x.reshape(-1, 1, np.size(x, 1), np.size(x, 2))
        y = [shmoo_title]
        self.X_train = torch.tensor(x, dtype=torch.float)
        self.Y_train = y  # np.array(y) #torch.tensor(y)
        pass

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


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()

        # This is pre-norm layer, may do not need it
        # self.input_norm = nn.Sequential(nn.BatchNorm2d(1))

        # First ConV layer, 2x2 kernel, input is 1 channel, output 32 channels, setp 1, padding = 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Pooling layer 3x3, step 2, no padding, but not used in this case
        # self.max_pool1 = nn.MaxPool2d(3, 2)

        # Second ConV layer, 3x3 kernel, input is 32 channels, output 16 channels, setp 1, padding = 1
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        # Third ConV layer, 3x3 kernel, input is 16 channels, output 8 channels, setp 1, padding = 1
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        # Fourth ConV layer, 3x3 kernel, input is 8 channels, output 4 channels, setp 1, no padding
        self.conv4 = nn.Sequential(
            nn.Conv2d(8, 4, 3, padding=0),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )

        # Pooling layer 3x3, step 2, no padding, but not used in this case
        # self.max_pool2 = nn.MaxPool2d(3, 2)

        # Fifth FC layer, input is 10x10x4, output is 64
        self.fc1 = nn.Sequential(
            nn.Linear(84, 42),
            nn.BatchNorm1d(42),
            nn.ReLU(),
            # nn.Dropout(0.2)
        )

        # Sixth FC layer, input is 64, output is 32
        self.fc2 = nn.Sequential(
            nn.Linear(42, 21),
            nn.BatchNorm1d(21),
            nn.ReLU(),
            # nn.Dropout(0.5)
        )

        # Seventh FC layer, input is 32, output is 6
        self.fc3 = nn.Sequential(
            nn.Linear(21, 6),
            # nn.BatchNorm1d(6),
            # nn.Sigmoid()
        )

    def forward(self, x):
        output_num = [4, 2, 1]
        # x = self.input_norm(x)
        x = self.conv1(x)
        # x = self.max_pool1(x)
        x = self.conv2(x)
        # x = self.max_pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.spatial_pyramid_pool(x, x.size(0), [int(x.size(2)), int(x.size(3))], output_num)

        # 将图片矩阵拉平
        # x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def spatial_pyramid_pool(self, previous_conv, num_sample, previous_conv_size, out_pool_size):
        '''
        previous_conv: a tensor vector of previous convolution layer
        num_sample: an int number of image in the batch
        previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
        out_pool_size: a int vector of expected output size of max pooling layer

        returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
        '''
        # print(previous_conv.size())
        for i in range(len(out_pool_size)):
            # print(previous_conv_size)
            h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
            w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
            h_pad = int(math.floor((h_wid * out_pool_size[i] - previous_conv_size[0] + 1) / 2))
            w_pad = int(math.floor((w_wid * out_pool_size[i] - previous_conv_size[1] + 1) / 2))
            # h_pad = min(h_pad, math.floor(h_wid / 2))
            # w_pad = min(w_pad, math.floor(w_wid / 2))
            # maxpool = nn.AvgPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
            maxpool = nn.AdaptiveMaxPool2d((out_pool_size[i], out_pool_size[i]))
            x = maxpool(previous_conv)
            if i == 0:
                spp = x.view(num_sample, -1)
                # spp = x.view(x.shape[0], -1)
                # print("spp size:",spp.size())
            else:
                # print("size:",spp.size())
                spp = torch.cat((spp, x.view(num_sample, -1)), 1)
                # spp = torch.cat((spp, x.view(x.shape[0], -1)), 1)
        return spp
