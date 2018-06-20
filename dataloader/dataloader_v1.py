import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class single_out_image():
    def __call__(self, sample):
        pass
class resize():
    def __call__(self, sample):
        pass
class zero_one():
    def __call__(self, sample):
        # sample = sample['state']
        sample = sample.astype(np.float) / 255.
        return sample
class numpy_pytorch_transpose():
    def __call__(self, sample):
        #numpy H W C
        #pytorch C H W
        sample = np.transpose(sample, (2, 0, 1))
        return sample
class Totensor():
    def __call__(self, sample):
        return torch.from_numpy(sample).float() # or long()

class Game_Frame(Dataset):

    def __init__(self, src_filename, transform=None):

        self.states = np.load(src_filename)['state']
        # self.states = np.load(src_filename)
        self.transform = transform

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        sample = self.states[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample
z_o = zero_one()
src_name = '/media/ray/SSD/workspace/python/dataset/save_here/rollout_0.npy'
# src_name = '/media/ray/SSD/workspace/python/dataset/save_here/rollout_v2_0.npz'
src_name = '/media/ray/SSD/workspace/python/dataset/save_here/rollout_tv2_0.npz'
# frame_dataset = Game_Frame(src_name)
frame_dataset = Game_Frame(src_name, transform=transforms.Compose(
    [
        zero_one(),
        numpy_pytorch_transpose(),
        Totensor(),
    ]
))

img = frame_dataset[9]
print(img.shape)
img = img.numpy()
print(img.shape)
img = np.transpose(img, (1, 2, 0))
print(img.shape)

print(img[10:15, 10:15, 2])
img = ((1-img) * 225).round().astype(np.uint8)
print(img[10:15, 10:15, 2])
# from PIL import Image
# img = Image.fromarray(img, 'RGB')
# img.show()
plt.imshow(img)
plt.show()
# print(frame_dataset[5].shape)
# print(frame_dataset[5][0, 10:15, 20:25])

# dataloader = DataLoader(frame_dataset, batch_size=4,
#                         shuffle=True, num_workers=4)
#
#
# for i, data in enumerate(dataloader):
#     frame, act = data['state'].size(), data['action']
#     print(act)
#     # print(frame[10:15, 10:15, 2])
#     if i == 1:
#         break


