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
class MDNRNN(nn.Module):
    def __init__(self, z_dim=32, action_dim=3, hidden_size=256, n_mixture=5, temperature=0.0):
        """
            :param z_dim: the dimension of VAE latent variable
            :param hidden_size: hidden size of RNN
            :param n_mixture: number of Gaussian Mixture Models to be used
            :param temperature: controls the randomness of the model
            MDNRNN stands for Mixture Density Network - RNN.
            The output of this model is [mean, sigma^2, K],
            where mean and sigma^2 have z_dim * n_mixture elements and
            K has n_mixture elements.
        """
        super(MDNRNN, self).__init__()
        # define rnn
        self.inpt_size = z_dim + action_dim
        self.hidden_size = hidden_size
        self.n_mixture = n_mixture
        self.z_dim = z_dim
        self.rnn = nn.LSTM(input_size=self.inpt_size, hidden_size=hidden_size, batch_first=True)

        # define MDN as fully connected layer
        self.mdn = nn.Linear(hidden_size, n_mixture * z_dim * 2 + n_mixture)
        self.tau = temperature

    def forward(self, inpt, action, hidden_state=None):
        """
        :param inpt: a tensor of size (batch_size, seq_len, D)
        :param hidden_state: two tensors of size (1, batch_size, hidden_size)
        :param action: a tensor of (batch_size, seq_len, action_dim)
        :return: pi, mean, sigma, hidden_state
        """
        batch_size, seq_len, _ = inpt.size()
        if hidden_state is None:
            # use new so that we do not need to know the tensor type explicitly.
            hidden_state = (Variable(inpt.data.new(1, batch_size, self.hidden_size)),
                            Variable(inpt.data.new(1, batch_size, self.hidden_size)))

        # concatenate input and action, maybe we can use an extra fc layer to project action to a space same
        # as inpt?
        concat = torch.cat((inpt, action), dim=-1)
        output, hidden_state = self.rnn(concat, hidden_state)
        output = output.contiguous()
        output = output.view(-1, self.hidden_size)
        # N, seq_len, n_mixture * z_dim * 2 + n_mixture
        mixture = self.mdn(output)
        mixture = mixture.view(batch_size, seq_len, -1)
        # N * seq_len, n_mixture * z_dim
        mean = mixture[..., :self.n_mixture * self.z_dim]
        sigma = mixture[..., self.n_mixture * self.z_dim: self.n_mixture * self.z_dim*2]
        sigma = torch.exp(sigma)
        # N * seq_len, n_mixture
        pi = mixture[..., -self.n_mixture:]
        pi = F.softmax(pi, -1)

        # add temperature
        if self.tau > 0:
            pi /= self.tau
            sigma *= self.tau ** 0.5
        return pi, mean, sigma, hidden_state
# help(cma)
# help(cma.CMAEvolutionStrategy)
env = CarRacing()
path = '/media/ray/SSD/workspace/python/dataset/save_here/model/'
vae_name = 'vae_model.save'
V = torch.load(path + vae_name)
V = V.cpu()
V.train(False)
rnn_name = 'rnn.save'
M = torch.load(path + rnn_name)
M.cpu()
M.train(False)
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
    # print(ee1)
    # print(ee2)
    # print(ee2>ee1)
    if ee1 > ee2:
        # print('here')
        # print(raw[1])
        # print(ee1)
        raw[1] = ee1
        # print(raw[1])
        raw[2]=0
    else:
        # print('oh')
        raw[1]=0
        raw[2]=ee2
    return raw

a = [-100., 5., 3.]
a = np.array(a)
a = a[None, :]
ac = action_ouput(a)
# print(ac)
es = cma.CMAEvolutionStrategy(para, 1, inopts={'popsize': 10})

# help(es)

def do_rollout(ctlr):
    done= False
    reward_sum = 0.
    state = env.reset()
    _ = env.render("rgb_array")     # render must come after reset
    m.lstm.reset()
    h = m.lstm.h_prev
    while not done:
        state = convert_frame(state)
        _,_,_,z = V(state[None,:])
        a = ctlr(z,h)
        state,reward,done,_ = env.step(a.data)
        reward_sum += reward
        az = torch.cat((a[None,:],z),dim=-1)
        _,h = m.lstm(az)
    return reward_sum

# a = np.array([0.0, 0.0, 0.0])

def get_a(z, h, para):
    num_z = 32
    num_h = 256
    input_size = num_z + num_h
    output_size = 3
    z_h = np.concatenate((z, h), axis=1)
    # print(z_h.shape)
    # para = np.random.random(input_size * output_size + output_size)
    # print(para.shape)
    # weights = np.random.random((input_size, output_size))
    weights = para[:input_size * output_size]
    weights = weights.reshape((input_size, output_size))

    # bias = np.random.random((1, output_size))
    bias = para[input_size * output_size:]
    a = np.matmul(ga, weights) + bias
    a = action_ouput(a)
    return a

def run_me(parameter):
    done = False
    reward_sum = 0.
    s = env.reset()
    env.render('rgb_array')
    a = np.array([0.0, 0.0, 0.0])
    counter = 0
    for _ in range(100):
        obs = state_to_1_batch_tensor(s)
        _, _, _, z = V(obs)
        a = torch.from_numpy(a).float()
        a = torch.reshape(a, (1, 1, 3))
        z = torch.reshape(z, (1, 1, 32))
        _, _, _, (h,c) = M(z, a)
        z = z.detach().numpy()
        h = h.detach().numpy()
        z = z.reshape((1, 32))
        h = h.reshape((1, 256))
        # print(type(z))
        # print(type(h))
        # print(len(h))
        # # print(h.size())
        # print(z.shape)
        # print(h.shape)
        # z = z.squeez()
        a = get_a(z, h, parameter)
        # print(a)
        s, reward, done, _ = env.step(a)
        # env.render()
        reward_sum += reward
        counter += 1
        print(counter)
    print(counter)
    return reward_sum



print(run_me(para))