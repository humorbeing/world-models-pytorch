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

dst = '/media/ray/SSD/workspace/python/dataset/save_here/rnn'
MAX_GAME_TIME = 1000
MAX_RUNS = 200
name_this = 'rnn_v0'
def generate_action():
    action = [0, 0, 0]
    action[0] = np.random.normal(0, np.random.random())
    acc_chance = 0.9
    if np.random.random() < acc_chance:
        action[1] = np.random.random()
        action[2] = 0
    else:
        action[1] = 0
        action[2] = np.random.random()
    return action


def _process_frame(frame):
    obs = frame[0:84, :, :].astype(np.float)/255.0
    obs = resize(obs, (64, 64))
    obs = ((1.0 - obs) * 255).round().astype(np.uint8)
    # obs = (obs * 255).round().astype(np.uint8)
    return obs


# REST_NUM = 250

import matplotlib.pyplot as plt
def multiple_runs(v, on):
    env = CarRacing()
    z_set = []
    action_set = []

    for run in range(MAX_RUNS):
        zs = []
        actions = []
        state = env.reset()
        env.render() # must have!
        # done = False
        counter = 0
        for game_time in range(MAX_GAME_TIME):
            # env.render()
            action = generate_action()
            obs = state_to_1_batch_tensor(state)
            _, _, _, z = v(obs)
            z = z.detach().numpy()
            z = z.reshape(32)
            # print(z.shape)
            # if game_time == 5:
            #     plt.imshow(state)
            #     plt.show()
            #     state = _process_frame(state)
            #     plt.imshow(state)
            #     plt.show()
            zs.append(z)
            actions.append(action)
            state, r, done, _ = env.step(action)

            # print(r)
            print(
                'RUN:{},GT:{},DATA:{}'.format(
                    run, game_time, len(actions)
                )
            )
            # if counter == REST_NUM:
            #
            #     position = np.random.randint(len(env.track))
            #     env.car = Car(env.world, *env.track[position][1:4])
            #     counter = 0
            # counter += 1
        zs = np.array(zs, dtype=np.float16)
        # print(zs.shape)
        actions = np.array(actions, dtype=np.float16)
        # print(actions.shape)

        # np.save(dst + '/' + save_name, frame_and_action)

        # np.savez_compressed(dst + '/' + save_name, action=actions, z=zs)
        z_set.append(zs)
        action_set.append(actions)
    z_set = np.array(z_set)
    # print(z_set.shape)
    action_set = np.array(action_set)
    # print(action_set.shape)
    save_name = name_this + '_{}.npz'.format(on)
    np.savez_compressed(dst + '/' + save_name, action=action_set, z=z_set)

if __name__ == '__main__':
    path = '/media/ray/SSD/workspace/python/dataset/save_here/model/'
    vae_name = 'vae_model.save'
    V = torch.load(path + vae_name)
    V = V.cpu()
    V.train(False)
    for i in range(10):
        multiple_runs(V, i)


