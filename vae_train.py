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
from vae_dataloader import vae_dataset_loader

EPOCH = 200
# src_name = '/media/ray/SSD/workspace/python/dataset/save_here/rollout_0.npy'
# src_name = '/media/ray/SSD/workspace/python/dataset/save_here/rollout_v2_0.npz'
src_name = '/media/ray/SSD/workspace/python/dataset/save_here/rollout_v3_0.npz'

dataloader = vae_dataset_loader(src_name, batch_size=512)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.nz = 32
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2),
            nn.ReLU())
        self.logvar_fc = nn.Linear(in_features=256 * 2 * 2, out_features=self.nz)
        self.mu_fc = nn.Linear(in_features=256 * 2 * 2, out_features=self.nz)

        self.decode_fc = nn.Linear(in_features=32, out_features=1024)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=6, stride=2),
            nn.Sigmoid())

        self._initialize_weights()

    def forward(self, x):
        vec = self.encoder(x)

        # flatten
        vec = vec.view(vec.size(0), -1)
        mu = self.mu_fc(vec)
        logvar = self.logvar_fc(vec)
        sigma = torch.exp(logvar/2.0)
        z = self.reparameterize(mu, sigma)
        im = self.decode_fc(z)

        # reshape into im
        im = im[:, :, None, None]

        xr = self.decoder(im)

        return xr, mu, logvar, z


    def decode_this(self, z):
        # pass
        im = self.decode_fc(z)

        # reshape into im
        im = im[:, :, None, None]

        xr = self.decoder(im)
        return xr

    def reparameterize(self, mu, sigma):
        if self.training:
            eps = Variable(torch.randn(*sigma.size()))
            if sigma.is_cuda:
                eps = eps.cuda()
            z = mu + eps * sigma
            return z
        else:
            return mu

    def _initialize_weights(self):
        # pass
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()



def vae_loss(x, recon_x, mu, logvar):
    rec = F.mse_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return rec + 0.000001*KLD.mean()


if __name__ == "__main__":
    path = '/media/ray/SSD/workspace/python/dataset/save_here/model/'
    vae_name = 'vae_model.save'
    is_new_model = False
    if is_new_model:
        V = VAE()
        V = V.cuda()
        V.train()
    else:
        V = torch.load(path + vae_name)
        V = V.cuda()
        V.train()

    optimizer = optim.Adam(V.parameters())

    for e in range(EPOCH):
        for batch_idx, state in enumerate(dataloader):

            x = Variable(state.cuda())
            optimizer.zero_grad()
            output, mu, logvar, z = V(x)
            # loss = F.mse_loss(output, x)
            # loss = F.binary_cross_entropy_with_logits(output, x)
            loss = vae_loss(x, output, mu, logvar)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    e, batch_idx * len(state), len(dataloader.dataset),
                           100. * batch_idx / len(dataloader),
                    # loss.data[0]
                    loss.item()
                ))
    torch.save(V, path + vae_name)