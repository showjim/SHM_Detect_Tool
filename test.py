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
        self.result_dict = {'Fail': [0], 'Pass': [1], 'Vol-Wall': [2], 'Freq-Wall': [3]}
        self.csv_df = pd.read_csv(csv_file, iterator=True, header=None)

        y = self.result_dict[self.csv_df.get_chunk(1).values[0][0]]
        x = self.csv_df.get_chunk(11).values

        self.X_train = torch.tensor(x, dtype=torch.int)
        self.Y_train = torch.tensor(y, dtype=torch.int)

    def __len__(self):
        # print len(self.landmarks_frame)
        # return len(self.landmarks_frame)
        return len(self.Y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.Y_train[idx]


filename = r'_new.csv'
dataset = CsvDataset(filename)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
for data in train_loader:
    print(data)