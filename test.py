# -*- coding: utf-8 -*-

import csv
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data


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
        x = np.zeros([11, 11], dtype=float)
        y = np.zeros([1], dtype=float)
        self.result_dict = {'Fail': [0.], 'Pass': [1.], 'Vol-Wall': [2.], 'Freq-Wall': [3.]}
        self.csv_df = pd.read_csv(csv_file, iterator=True, header=None)
        # Read data in chunck
        go = True
        while go:
            try:
                tmp_y = self.result_dict[self.csv_df.get_chunk(1).values[0][0]]
                tmp_x = self.csv_df.get_chunk(11).values
                y = np.vstack((y, tmp_y))
                x = np.vstack((x, tmp_x))
                self.data_count += 1
            except Exception as e:
                print(type(e))
                go = False
        # Reshape the data
        y = y.reshape(-1, 1, 1)
        x = x.reshape(-1, 11, 11)

        self.X_train = torch.tensor(x, dtype=torch.float)
        self.Y_train = torch.tensor(y, dtype=torch.float)

    def __len__(self):
        # print len(self.landmarks_frame)
        # return len(self.landmarks_frame)
        return len(self.Y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.Y_train[idx]


filename = r'_new.csv'
dataset = CsvDataset(filename)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=dataset.data_count, shuffle=True)
for data in train_loader:
    print(data)
