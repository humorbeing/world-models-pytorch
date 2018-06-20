import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import gym
import matplotlib.pyplot as plt
import torch

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
z_o = zero_one()
# src_name = '/media/ray/SSD/workspace/python/dataset/save_here/rollout_0.npy'
# src_name = '/media/ray/SSD/workspace/python/dataset/save_here/rollout_v2_0.npz'
src_name = '/media/ray/SSD/workspace/python/dataset/save_here/rollout_v3_0.npz'
# frame_dataset = Game_Frame(src_name)
frame_dataset = Game_Frame(src_name, transform=transforms.Compose(
    [
        zero_one(),
        numpy_pytorch_transpose(),
        Totensor(),
    ]
))
dataloader = DataLoader(frame_dataset, batch_size=128,
                        shuffle=True, num_workers=4)


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


def vae_loss(x, xr, mu, logvar):
    rec = F.mse_loss(xr, x)
    # print(rec.shape)
    # print('mu:',mu.shape)
    # print('sigma:',sigma.shape)
    # mu_sum_sq = (mu * mu).sum(dim=1)
    # sig_sum_sq = (sigma * sigma).sum(dim=1)
    # log_term = (1 + torch.log(sigma ** 2)).sum(dim=1)
    # kldiv = -0.5 * (log_term - mu_sum_sq - sig_sum_sq)
    kl = - 0.5 * (1 + logvar - mu*mu - torch.exp(logvar))
    # print(kl.shape)
    kl = torch.sum(kl, dim=1)
    # print(kl.shape)
    # print(z.shape)
    # kl = torch.max(kl, 0.5*32)
    kl = torch.mean(kl)
    # print(kl.shape)
    return rec + kl


if __name__ == "__main__":
    V = VAE().cuda()
    V.train()
    # all_actions = np.zeros((args.rollouts, rollout_len + 1, len_action))
    # all_z = np.zeros((args.rollouts, rollout_len + 1, V.nz))
    # az_fn = os.path.join(az_pair_dir, "az.npz")

    optimizer = optim.Adam(V.parameters())
    # criterion = vae_loss
    for e in range(200):
        for batch_idx, state in enumerate(dataloader):
            # state = data
            x = Variable(state.cuda())
            optimizer.zero_grad()
            output, mu, logvar, z= V(x)
            # loss = F.binary_cross_entropy_with_logits(output, x)
            loss = vae_loss(x, output, mu, logvar)
            loss.backward()
            optimizer.step()
            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    e, batch_idx * len(state), len(dataloader.dataset),
                           100. * batch_idx / len(dataloader),
                    # loss.data[0]
                    loss.item()
                ))
