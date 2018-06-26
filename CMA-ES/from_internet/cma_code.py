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
from torchvision.transforms import Compose,Normalize,Resize,ToTensor
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import math
from torchvision.utils import make_grid
from pyglet.window import key
from vae_dataloader import state_to_1_batch_tensor
from vae_dataloader import one_batch_tensor_to_img
from vae_train import VAE

class Controller(nn.Module):
    def __init__(self, parameters=None):
        super(Controller, self).__init__()
        self.fc = nn.Linear(32, 3)
        if parameters is not None:
            pass
    def forward(self, x):
        return x

c = Controller()
# c.train(False)
c.eval()
print(c.training)
# fc = nn.Linear(32, 3)
# a = fc.weight.size()
# print(a)
# b = np.prod([2,3])
# print(b)
# c = fc.bias.size()
# print(c)
#
# d = torch.rand(2,1,3)
# print(d.shape)
# d = d.squeeze()
# print(d.shape)