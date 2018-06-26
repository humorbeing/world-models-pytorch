# import os
import torch
# import pandas as pd
# from skimage import io, transform
import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class RNN_DATASET(Dataset):

    def __init__(self, src_filename, transform=None):

        self.actions = np.load(src_filename)['action']
        self.zs = np.load(src_filename)['z']
        self.transform = transform

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        a = self.actions[idx]
        z = self.zs[idx]
        # sample = self.states[idx]
        if self.transform:
            # pass
            a = self.transform(a)
            z = self.transform(z)

        return a, z

class Totensor():
    def __call__(self, sample):
        return torch.from_numpy(sample).float()




def rnn_dataset_loader(src_name, batch_size, shuffle=True, num_workers=4):
    # src_name = '/media/ray/SSD/workspace/python/dataset/save_here/rnn/rnn_v0_10_1.npz'
    dataset = RNN_DATASET(src_name, transform=transforms.Compose(
        [
            # zero_one(),
            # numpy_pytorch_transpose(),
            Totensor(),
        ]
    ))

    # a, z = dataset[0]
    # print(a.shape)
    # print(z.shape)
    # print(a[1])
    # print(z[1])

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


# src_name = '/media/ray/SSD/workspace/python/dataset/save_here/rnn/rnn_v0_10_1.npz'
# d = vae_dataset_loader(src_name, 9)
#
# for bat, data in enumerate(d):
#     a, z = data
#     print(a.shape)
#     break