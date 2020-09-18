# -*- coding: utf-8 -*-
import sys
import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import torch.utils.data as data
import torch.nn.functional as F


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
    acc_sum, n, acc_sum_pf = 0.0, 0, 0.0
    for X, y in data_iter:
        # acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        y_hat = net(X)
        y_hat[y_hat > 0.5] = 1
        y_hat[y_hat <= 0.5] = 0
        check_result = torch.sum(y_hat == y, 1)
        check_result_pf = (y_hat[0] == y[0]) and (y_hat[1] == y[1])
        acc_sum += (check_result == 6).float().sum().item()
        acc_sum_pf += check_result_pf
        n += y.shape[0]
    return acc_sum / n, acc_sum_pf / n


def sgd(params, lr, batch_size):  # d2lzh_pytorch
    for param in params:
        param.data -= lr * param.grad / batch_size


def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
    plt.ion()
    fig = plt.figure()
    plt.axis([0, 200, 0., 1.])
    plt.grid()
    for epoch in range(num_epochs):
        # adjust_learning_rate(optimizer, epoch, lr)
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
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
            y_hat[y_hat > 0.5] = 1
            y_hat[y_hat <= 0.5] = 0
            check_result = torch.sum(y_hat == y, 1)
            train_acc_sum += (check_result == 6).sum().item()
            n += y.shape[0]
        test_acc, test_acc_pf = evaluate_accuracy(test_iter, net)
        # plt.clf()
        plt.plot(epoch, test_acc, '.r')
        # plt.grid()
        # plt.plot(epoch, train_l_sum / n, 'xb')
        plt.pause(0.0001)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, test acc P/F %.3f' % (
            epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc, test_acc_pf))
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

def show_fashion_mnist(images, labels):
    # use_svg_display()
    _, figs = plt.subplots(5, int(len(images) / 5), figsize=(12, 8))
    plt.tight_layout()
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
        y = np.zeros([6], dtype=float)
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
        #                     'Fail-Marginal-Hole':     [1, 0, 0, 0, 1, 1]}
        self.csv_df = pd.read_csv(csv_file, iterator=True, header=None)
        # Read data in chunck
        go = True
        while go:
            try:
                tmp_y = self.convert_result(self.csv_df.get_chunk(1).values[0][0])
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

    def convert_result(self, text):
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
        x = x.reshape(-1, 1, 11, 11)

        self.X_train = torch.tensor(x, dtype=torch.float)
        self.Y_train = y #np.array(y) #torch.tensor(y)
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


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 3, padding=(1, 1)),  # in_channels, out_channels, kernel_size, padding
            nn.ReLU(),
            # nn.MaxPool2d(2, 2),  # kernel_size, stride
            nn.Conv2d(6, 16, 3, padding=(1, 1)),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 11 * 11, 80),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(80, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    blk = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size=1),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size=1),
                        nn.ReLU()
                        )
    return blk


class Inception(nn.Module):
    # c1 - c4 are the channel number of each line
    def __init__(self, in_c, c1, c2, c3, c4):
        super(Inception, self).__init__()
        # line 1: 1 x 1 conv
        self.p1_1 = nn.Conv2d(in_c, c1, kernel_size=1)
        # line 2: 1 x 1 conv then 3 x 3 conv
        self.p2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # line3: 1 x 1 conv then 5 x 5 conv
        self.p3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # line 4: 3 x 3 pool then 1 x 1 conv
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_c, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)  # Concatenates the given sequence of seq tensors


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 第一层是 5x5 的卷积，输入的channels 是 3，输出的channels是 64,步长 1,没有 padding
        # Conv2d 的第一个参数为输入通道，第二个参数为输出通道，第三个参数为卷积核大小
        # ReLU 的参数为inplace，True表示直接对输入进行修改，False表示创建新创建一个对象进行修改
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 2, padding=1),
            nn.ReLU()
        )

        # 第二层为 3x3 的池化，步长为2，没有padding
        # self.max_pool1 = nn.MaxPool2d(3, 2)

        # 第三层是 5x5 的卷积， 输入的channels 是64，输出的channels 是64，没有padding
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(8, 4, 3, padding=0),
            nn.ReLU()
        )

        # 第四层是 3x3 的池化， 步长是 2，没有padding
        # self.max_pool2 = nn.MaxPool2d(3, 2)

        # 第五层是全连接层，输入是 1204 ，输出是384
        self.fc1 = nn.Sequential(
            nn.Linear(10 * 10 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 第六层是全连接层，输入是 384， 输出是192
        self.fc2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # 第七层是全连接层，输入是192， 输出是 10
        self.fc3 = nn.Sequential(
            nn.Linear(32, 6)
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        # x = self.max_pool1(x)
        x = self.conv2(x)
        # x = self.max_pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # 将图片矩阵拉平
        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
