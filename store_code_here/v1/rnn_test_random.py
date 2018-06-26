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
from rnn_train import MDNRNN


def show_state(s, step=0):
  plt.figure(3)
  plt.clf()
  plt.imshow(255-s)
  # plt.title("%s. Step: %d" % (env._spec.id, step))

  plt.pause(0.001)  # pause for plots to update
# from https://github.com /openai/gym/blob/master/gym/envs/box2d/car_racing.py


if __name__ == "__main__":
    path = '/media/ray/SSD/workspace/python/dataset/save_here/model/'
    vae_name = 'vae_model.save'
    V = torch.load(path + vae_name)
    V = V.cpu()
    V.train(False)
    rnn_name = 'rnn.save'
    M = torch.load(path + rnn_name)
    M.cpu()
    M.train(False)

    a = np.array([0.0, 0.0, 0.0])






    while True:
        # s = env.reset()
        s = np.zeros((96, 96, 3))
        # print(s.shape)
        # input('wait')
        obs = state_to_1_batch_tensor(s)
        for _ in range(1000):
            _, _, _, z = V(obs)
            a = torch.from_numpy(a).float()
            a = torch.reshape(a, (1,1,3))
            z = torch.reshape(z, (1,1,32))
            # concat = torch.cat((z,a), dim=-1)
            # print(concat.shape)
            # print(a.shape)
            # print(z.shape)
            pi, mean, sigma, hidden_state = M(z, a)
            # print(pi.shape)
            # hidden_state = hidden_state.detach().numpy()
            pi = pi.detach().numpy()
            mean = mean.detach().numpy()
            sigma = sigma.detach().numpy()
            pi = pi.reshape(5)
            sigma = sigma.reshape(160)
            mean = mean.reshape(160)
            print(pi)
            print(np.max(pi))
            print(np.argmax(pi))
            mmm = np.argmax(pi)
            z_pre = np.zeros(32)

            print()
            print(mmm)
            print(mean.shape)
            print(z_pre.shape)
            for i in range(32):
                z_pre[i] = mean[(i*5+mmm)]
            print(z_pre)
            # aa, bb = torch.max(pi, 2)
            # print(aa, bb)
            # out = Variable(torch.zeros(1,1,))
            z_pre = z_pre.reshape((1, 32))
            z_pre = torch.from_numpy(z_pre).float()
            print(z_pre.shape)
            xr = V.decode_this(z_pre)
            print(xr.shape)
            xr = xr.detach().numpy()
            img = one_batch_tensor_to_img(xr)
            show_state(img)
            print(z.shape)
            z_pre = torch.reshape(z_pre, (1,1,32))
            print(z_pre.shape)

            z = z_pre
            # input('wait')
            a = np.random.random(3)
    #     total_reward = 0.0
    #     steps = 0
    #     restart = False
    #     # env.render()
    #     while True:
    #         s, r, done, info = env.step(a)
    #         total_reward += r
    #         if steps % 200 == 0 or done:
    #             # print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
    #             # print("step {} total_reward {:+0.2f}".format(steps, total_reward))
    #             pass
    #         steps += 1
    #         env.render()
    #
    #         # img = rec.detach().numpy()
    #         img = one_batch_tensor_to_img(img)
    #         show_state(img)
    #         if done or restart: break
    # env.monitor.close()

