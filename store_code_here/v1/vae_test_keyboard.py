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

    a = np.array([0.0, 0.0, 0.0])


    def key_press(k, mod):
        global restart
        if k == 0xff0d: restart = True
        if k == key.LEFT:  a[0] = -1.0
        if k == key.RIGHT: a[0] = +1.0
        if k == key.UP:    a[1] = +1.0
        if k == key.DOWN:  a[2] = +0.8  # set 1.0 for wheels to block to zero rotation


    def key_release(k, mod):
        if k == key.LEFT and a[0] == -1.0: a[0] = 0
        if k == key.RIGHT and a[0] == +1.0: a[0] = 0
        if k == key.UP:    a[1] = 0
        if k == key.DOWN:  a[2] = 0


    env = CarRacing()
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    while True:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        env.render()
        while True:
            s, r, done, info = env.step(a)
            total_reward += r
            if steps % 200 == 0 or done:
                # print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                # print("step {} total_reward {:+0.2f}".format(steps, total_reward))
                pass
            steps += 1
            env.render()
            obs = state_to_1_batch_tensor(s)
            rec,_,_,_ = V(obs)
            img = rec.detach().numpy()
            img = one_batch_tensor_to_img(img)
            show_state(img)
            if done or restart: break
    env.monitor.close()