import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class zero_one():
    def __call__(self, sample):
        action, state = sample['action'], sample['state']
        state = state / 255
        return {'action': action, 'state': state}

class Game_Frame(Dataset):

    def __init__(self, src_filename, transform=None):

        self.frame_and_action = np.load(src_filename)
        self.transform = transform

    def __len__(self):
        return len(self.frame_and_action)

    def __getitem__(self, idx):
        sample = self.frame_and_action[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample
z_o = zero_one()
src_name = '/media/ray/SSD/workspace/python/dataset/save_here/rollout_0.npy'
# frame_dataset = Game_Frame(src_name)
frame_dataset = Game_Frame(src_name, transform=transforms.Compose(
    [
        z_o
    ]
))
print(len(frame_dataset))
for i in range(5):
    print(frame_dataset[i]['action'])
    # print(frame_dataset[])
print(frame_dataset[7]['state'][10:15,10:15,2])
dataloader = DataLoader(frame_dataset, batch_size=4,
                        shuffle=True, num_workers=4)


for i, data in enumerate(dataloader):
    frame, act = data['state'].size(), data['action']
    print(act)
    # print(frame[10:15, 10:15, 2])
    if i == 1:
        break


