# import os
import torch
# import pandas as pd
# from skimage import io, transform
import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class zero_one():
    def __call__(self, sample):
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
        self.transform = transform

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        sample = self.states[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample
# z_o = zero_one()
# src_name = '/media/ray/SSD/workspace/python/dataset/save_here/rollout_0.npy'
src_name = '/media/ray/SSD/workspace/python/dataset/save_here/rollout_v2_0.npz'
# frame_dataset = Game_Frame(src_name)
frame_dataset = Game_Frame(src_name, transform=transforms.Compose(
    [
        zero_one(),
        numpy_pytorch_transpose(),
        Totensor(),
    ]
))

def vae_dataset_loader(src_name, batch_size, shuffle=True, num_workers=4):

    frame_dataset = Game_Frame(src_name, transform=transforms.Compose(
        [
            zero_one(),
            numpy_pytorch_transpose(),
            Totensor(),
        ]
    ))
    dataloader = DataLoader(frame_dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers)
    return dataloader


_zero = zero_one()
_numpy = numpy_pytorch_transpose()
_t = Totensor()
from scipy.misc import imresize as resize
def _process_frame(frame):
  obs = frame[0:84, :, :].astype(np.float)/255.0
  obs = resize(obs, (64, 64))
  obs = ((1.0 - obs) * 255).round().astype(np.uint8)
  return obs

def state_to_1_batch_tensor(s):
    obs = _process_frame(s)
    obs = _zero(obs)
    obs = _numpy(obs)
    obs = _t(obs)
    # print(obs.shape)
    obs = obs[None, :, :, :]
    # print(obs.shape)
    return obs


def one_batch_tensor_to_img(img):
    img = img.reshape(3, 64, 64)
    # print(s[20:23, 20:23, 1])
    # print(img[1,20:23, 20: 23])
    # print(img.shape)
    img = np.transpose(img, (1, 2, 0))
    img = ((1 - img) * 225).round().astype(np.uint8)
    # print(s[20:22, 20:22, 1])
    # print(img[20:22, 20:22, 1])
    # print()
    return img