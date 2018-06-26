import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import gym
import matplotlib.pyplot as plt
import torch
from scipy.misc import imresize as resize
from gym.spaces.box import Box
from gym.envs.box2d.car_racing import CarRacing
from matplotlib import pyplot as plt
# setup rendering before importing other stuff (weird hack to avoid pyglet errors)
# env = gym.make("CarRacing-v0")
# _ = env.reset()
# _ = env.render("rgb_array")
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import init
from torch import optim
import os
import time
# from tensorboardX import SummaryWriter
import sys
import torchvision
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import math
from torchvision.utils import make_grid
from pyglet.window import key
from vae_dataloader import state_to_1_batch_tensor
from vae_dataloader import one_batch_tensor_to_img
from vae_train import VAE
import cma
# help(cma)
# help(cma.CMAEvolutionStrategy)

num_z = 32
num_h = 256
input_size = num_z + num_h
output_size = 3
para = np.random.random(input_size*output_size+output_size)
print(para.shape)
# weights = np.random.random((input_size, output_size))
weights = para[:input_size*output_size]
weights = weights.reshape((input_size, output_size))

# bias = np.random.random((1, output_size))
bias = para[input_size*output_size:]

print(weights.shape)
print(bias.shape)
ga = np.zeros((1, 288))
a = np.matmul(ga, weights) + bias
print(a)

def action_ouput(raw):
    raw = raw.reshape(3)
    # print(raw)
    raw[0] = np.tanh(raw[0])
    e1 = np.exp(raw[1])
    e2 = np.exp(raw[2])
    ee = e1 + e2
    ee1 = e1 / ee
    ee2 = e2 / ee
    print(ee1)
    print(ee2)
    print(ee2>ee1)
    if ee1 > ee2:
        print('here')
        print(raw[1])
        print(ee1)
        raw[1] = ee1
        print(raw[1])
        raw[2]=0
    else:
        print('oh')
        raw[1]=0
        raw[2]=ee2
    return raw

a = [-100., 5., 3.]
a = np.array(a)
a = a[None, :]
ac = action_ouput(a)
print(ac)