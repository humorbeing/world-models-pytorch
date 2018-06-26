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

from rnn_dataloader import rnn_dataset_loader

# src = '/media/ray/SSD/workspace/python/dataset/save_here/rnn/rnn_v0_10_1.npz'
# src = '/media/ray/SSD/workspace/python/dataset/save_here/rnn/rnn_v0_0.npz'
src = '/media/ray/SSD/workspace/python/dataset/save_here/rnn/rnn_v0_1.npz'
rnn_dataset = rnn_dataset_loader(src, batch_size=10)
is_cuda = torch.cuda.is_available()

class LSTM(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers)

        self.hidden_size = hidden_size

        self.batch_size = batch_size
        self.h_prev = None
        self.c_prev = None
        self.reset()

    def reset(self):
        self.h_prev = Variable(torch.Tensor(1, self.batch_size, self.hidden_size).normal_())  # .cuda()
        self.c_prev = Variable(torch.Tensor(1, self.batch_size, self.hidden_size).normal_())  # .cuda()
        if is_cuda:
            self.h_prev = self.h_prev.cuda()
            self.c_prev = self.c_prev.cuda()

    def forward(self, az):
        lstm_out, (self.h_prev, self.c_prev) = self.rnn(az[None, :], (self.h_prev, self.c_prev))

        return lstm_out, self.h_prev


class MDN(nn.Module):
    def __init__(self, input_size, output_size, num_gaussians, nz, temperature):
        super(MDN, self).__init__()

        self.nz = nz
        self.num_gaussians = num_gaussians
        self.mdn_fc = nn.Linear(in_features=input_size,
                                out_features=output_size)
        self.temperature = temperature

    def postproc_mdn_out(self, mdn_out):
        mu = mdn_out[:, :self.num_gaussians * self.nz]

        sigma = mdn_out[:, self.num_gaussians * self.nz:2 * self.num_gaussians * self.nz]

        pi = mdn_out[:, -self.num_gaussians:]

        mu = mu.resize(mu.size(0), self.num_gaussians, self.nz)

        sigma = torch.exp(sigma)
        sigma = sigma.resize(sigma.size(0), self.num_gaussians, self.nz)
        pi = self.temperature * pi
        pi = F.softmax(pi, dim=1)
        return mu, sigma, pi

    def forward(self, lstm_out):
        raw_mdn_out = self.mdn_fc(lstm_out)
        mu, sigma, pi = self.postproc_mdn_out(raw_mdn_out)
        return mu, sigma, pi


class MM(nn.Module):
    def __init__(self, batch_size, env="CarRacing", num_gaussians=5, num_layers=1, temperature=1.):
        super(MM, self).__init__()
        if env == "CarRacing":
            self.nz = 32
            self.nh = 256
            self.action_len = 3  # 3 continuous values
        elif env == "Doom":
            pass  # self.nz, self.nh = 64, 512

        self.batch_size = batch_size
        self.temperature = temperature
        self.num_gaussians = num_gaussians
        self.mu_len, self.sigma_len, self.pi_len = self.nz, self.nz, 1

        self.len_mdn_output = self.num_gaussians * (self.sigma_len + self.mu_len + self.pi_len)

        self.lstm = LSTM(batch_size=self.batch_size,
                         input_size=self.nz + self.action_len,
                         hidden_size=self.nh,
                         num_layers=num_layers)

        self.mdn = MDN(input_size=self.nh,
                       output_size=self.len_mdn_output,
                       num_gaussians=self.num_gaussians,
                       nz=self.nz, temperature=self.temperature)

    def forward(self, a, z):
        self.lstm.reset()
        # print(a.size(), z.size())
        az = torch.cat((z, a), dim=-1)
        mus, sigmas, pis, hs = [], [], [], []

        for azi in az:
            lstm_out, h = self.lstm(azi)
            mu, sigma, pi = self.mdn(lstm_out[0])

            mus.append(mu[None, :])
            sigmas.append(sigma[None, :])
            pis.append(pi[None, :])
            hs.append(h[None, :])
        mus = torch.cat(mus)
        sigmas = torch.cat(sigmas)
        pis = torch.cat(pis)
        hs = torch.cat(hs)

        return mus, sigmas, pis, hs
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
    # def reset_h(self):
    #     hidden_state = (Variable(inpt.data.new(1, batch_size, self.hidden_size)),
    #                     Variable(inpt.data.new(1, batch_size, self.hidden_size)))
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
def gaussian_pdf(x, mu, sigmasq):
  # NOTE: we could use the new `torch.distributions` package for this now
  # a = torch.sqrt(2*np.pi*sigmasq)
  # print(a.shape)
  # b = -1 / (2 * sigmasq)
  # # b = torch.exp(()* torch.norm((x-mu), 2, 1)**2)
  # print(b.shape)
  # # c = torch.norm((), 2, 1)**2
  # c = x-mu
  # print(c.shape)
  # d = torch.norm(c, 2, dim=2, keepdim=True)**2
  # print('d:',d.shape)
  # e = c*d
  # print(e.shape)
  # f = torch.exp(e)
  # print(f.shape)
  return (1/torch.sqrt(2*np.pi*sigmasq)) * torch.exp((-1/(2*sigmasq)) * torch.norm((x-mu), 2, dim=2, keepdim=True)**2)

def mdn_criterion(mus,sigmas,pis,label):
    # print('mus:', mus.shape)
    # print('sigmas:', sigmas.shape)
    # print('pis:', pis.shape)
    # print('label', label.shape)
    # z is batch of seq_len number of z's, where each z is nz long
    #print(z.size())
    # mus is for each element in the seqence, a batch of a mixture of num_guasians mean vectors, where each mean vector has nz elements
    #print(mus.size())
    # sigmas is for each element in the sequence,  a batch of a mixture of num_guassians covariance vectors, where each covariance vector has nz elements
    # and represents the diagonal of the covariance matrix
    #print(sigmas.size())
    # we want to compute the density of a batch of z's under this batch of mixtures
    # pad z with a dummy dimension to enable broadcasting over the num_mixture_components dimension
    # label = torch.unsqueeze(label,dim=2)
    k = 5
    n = 32
    # print()
    # a = label.size()
    # print(a)
    # input('wait')
    losses = Variable(torch.zeros((label.size()))).cuda()
    # print(mus[:, :, 0 * n:(0 + 1) * n].shape)
    # print(sigmas[:, :, 1 * n:(1 + 1) * n].shape)
    for i in range(k):
        likelihood_z_x = gaussian_pdf(label, mus[:, :, i * n:(i + 1) * n], sigmas[:, :, i * n:(i + 1) * n])
        prior_z = pis[:, :, i]
        prior_z = prior_z[:,:,None]
        # print('pp')
        # print(prior_z.shape)
        # print(likelihood_z_x.shape)
        losses += prior_z * likelihood_z_x
    loss = torch.mean(-torch.log(losses))
    return loss
    '''
    print('label:', label.shape)
    #print(z.size())
    # we parametrize a normal distribution for every element in the sequence for every example in the batch for every mixture for every dimension
    nd = torch.distributions.Normal(mus,sigmas)
    print(type(nd))
    # print('nd:',nd.shape)
    # because the covariance matrix is diagonal the probability if z under a given mean vector and cov matrix is the product
    # of the density of each element of z under a univariate guassian. For log prob, this turns into a sum. So
    # if we sum in the dimension of the elements of z, then we get the log density of each sequence index for each example under each mixture
    log_prob_elwise = nd.log_prob(label)
    #print(log_prob_elwise.size())
    log_prob = log_prob_elwise.sum(dim=-1)
    #print(log_prob.size())
    # pis is number of examples by mixture coefficients, so we can just elementwise multoply this with log_probs
    # and sum along the mixture component direction
    #print(pis.size())
    NLL = -(pis * log_prob).sum(dim=-1)
    # now we have negative log likelihood for each element in the sequence for each example in the batch
    #print(NLL.size())

    # now lets sum over each element in the sequence
    seq_NLL = NLL.sum(dim=0)
    #print(seq_NLL.size())
    # now we take the mean over the batch
    loss = seq_NLL.mean()
    #print(loss.size())
    return loss[0]
    '''

if __name__ == "__main__":
    # az_f = np.load(args.az_file)
    #
    # a, z = az_f["a"], az_f["z"]
    #
    # a = a[:, :-1]
    #
    # azds = AZDataset(a, z)
    #
    # azdl = DataLoader(azds, shuffle=True, batch_size=args.batch_size)
    dst = '/media/ray/SSD/workspace/python/dataset/save_here/model/rnn.save'
    # torch.save(M, dst)
    M = torch.load(dst)
    # M.lstm.reset()
    if is_cuda:
        M = M.cuda()
    optimizer = optim.Adam(M.parameters())
    for epoch in range(20):
        # torch.save(m.state_dict(), '%s/curr_%s.pth' % (saved_model_dir, basename))
        losses = []
        for it, (a, z) in enumerate(rnn_dataset):
            a = a[:, :-1, :]
            zin = z[:, :-1, :]
            label = z[:, 1:, :]
            a = Variable(a.cuda())
            zin = Variable(zin.cuda())
            label = Variable(label.cuda())
            # a.transpose_(1, 0)
            # z.transpose_(1, 0)
            # a = Variable(a).float()
            # zinp = Variable(z[:-1]).float()
            # # our label is the NEXT frame in the sequence, so the az that is input is matched with the next frame down
            # # so we don't need the first frame for our labels
            # label = Variable(z[1:]).float()
            # if is_cuda:
            #     zinp = zinp.cuda()
            #     a = a.cuda()
            #     label = label.cuda()
            M.zero_grad()
            # we will have one more frame than action because we don't take an action after the last frame
            # here we push the az's through the rnn to get parameters of a mixture of guassians
            # we don't throw the last z in there b/c it has no action for it
            pis, mus, sigmas, hs = M(a, zin)
            # print(hs)
            # print(hs.detach().numpy().shape)
            # input('wait')
            loss = mdn_criterion(mus, sigmas, pis, label)
            # losses.append(loss.data[0])

            # print(loss.data[0])
            # writer.add_scalar("iter_loss", loss.data[0], global_step=epoch * len(azds) + it)
            loss.backward()
            optimizer.step()
            print(loss)
        # torch.save(m.state_dict(), '%s/curr_%s.pth' % (saved_model_dir, basename))
        # if epoch % 20 == 0:
        #     torch.save(m.state_dict(), '%s/curr_%s_%s.pth' % (saved_model_dir, basename, epoch))
        # writer.add_scalar("loss", np.mean(losses), global_step=epoch)
    dst = '/media/ray/SSD/workspace/python/dataset/save_here/model/rnn.save'
    torch.save(M, dst)