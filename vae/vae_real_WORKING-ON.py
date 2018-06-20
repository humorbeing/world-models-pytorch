import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class zero_one():
    def __call__(self, sample):
        sample = np.asarray(sample)
        sample = sample.astype(np.float) / 255.
        # print(sample.shape)
        sample = sample[:,:,None]
        # print(sample.shape)
        return sample
class numpy_pytorch_transpose():
    def __call__(self, sample):
        #numpy H W C
        #pytorch C H W
        sample = np.transpose(sample, (2, 0, 1))
        return sample
class Totensor():
    def __call__(self, sample):
        return torch.from_numpy(sample).float()
transf = transforms.Compose([
                       zero_one(),
                       numpy_pytorch_transpose(),
                       Totensor()
                       # transforms.Normalize((0.1307,), (0.3081,))
                       # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                   ])
# dst = '/media/ray/D43E51303E510CBC/MyStuff/Workspace/Python/dataset/original/mnist'
dst = '/media/ray/SSD/workspace/python/dataset/original/mnist'
mnist_trainset = datasets.MNIST(
    dst,
    train=True,
    download=True,
    transform=transf
)
mnist_testset = datasets.MNIST(
    dst,
    train=False,
    download=True,
    transform=transf
)


train_loader = torch.utils.data.DataLoader(
    mnist_trainset,
    batch_size=128,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    mnist_testset,
    batch_size=10,
    shuffle=False
)
gpu = torch.cuda.is_available()


def to_img(x):
    x = x.data.numpy()
    x = 0.5 * (x + 1)
    x = np.clip(x, 0, 1)
    x = x.reshape([-1, 28, 28])
    return x


def plot_reconstructions(model, save=False, name=None, conv=False, simple=False):
    """
    Plot 10 reconstructions from the test set. The top row is the original
    digits, the bottom is the decoder reconstruction.
    """
    # encode then decode
    data, _ = next(iter(test_loader))
    # data, _ = next(iter(test_loader))
    if not conv:
        data = data.view([-1, 784])
    data = Variable(data)#, volatile=True)
    true_imgs = data
    # model.training = False
    model.train(False)
    xr,_,_,_ = model(data)
    # if simple:
    #     encoded_imgs = F.relu(encoded_imgs)
    # decoded_imgs = model.decoder(encoded_imgs)

    true_imgs = to_img(true_imgs)
    decoded_imgs = to_img(xr)

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(true_imgs[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    if save:
        plt.savefig('./' + name + '.png', format='png', dpi=300)
    plt.show()

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 9),
            nn.ReLU(True),
            nn.Conv2d(16, 16, 9),
            nn.ReLU(True),
            nn.Conv2d(16, 16, 9),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(True),
        )
        self.logvar_fc = nn.Linear(2*2*32, 32)
        self.mu_fc = nn.Linear(2*2*32, 32)
        self.decode_fc = nn.Linear(32, 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 16, 5),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 16, 9),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 4, 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(4, 1, 9),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, sigma):
        if self.training:
            eps = Variable(torch.randn(*sigma.size())).cuda()
            # if sigma.is_cuda:
            #     eps = eps.cuda()
            z = mu + eps*sigma
            return z
        else:
            return mu

    def forward(self, x):
        vec = self.encoder(x)
        vec = vec.view(vec.size(0), -1)
        mu = self.mu_fc(vec)
        logvar = self.logvar_fc(vec)
        sigma = torch.exp(logvar / 2.0)
        z = self.reparameterize(mu, sigma)
        im = self.decode_fc(z)
        im = im[:, :, None, None]
        xr = self.decoder(im)
        return xr, mu, logvar, z


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

input_dim = 784
encoding_dim = 10

model = AutoEncoder()
model.cuda()
optimizer = optim.Adam(model.parameters())


def l1_penalty(var):
    return torch.abs(var).sum()


def train(epoch, sparsity=False, l1_weight=1e-5):
    for batch_idx, (data, _) in enumerate(train_loader):
        # data = Variable(data.view([-1, 784]).cuda())
        data = Variable(data.cuda())
        optimizer.zero_grad()

        # enforcing sparsity with l1 reg
        if sparsity:
            decoder_out, encoder_out = model(data)
            mse_loss = F.mse_loss(decoder_out, data)
            l1_reg = l1_weight * l1_penalty(encoder_out)
            loss = mse_loss + l1_reg
        else:
            xr, mu, logvar, z = model(data)
            loss = vae_loss(data, xr, mu, logvar)
            # loss = F.binary_cross_entropy_with_logits(xr, data)
            # loss = F.mse_loss(output, data)

        loss.backward()
        optimizer.step()
        # print(epoch)
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                # loss.data[0]
                loss.item()
            ))


num_epochs = 10

for epoch in range(1,  num_epochs + 1):
    train(epoch)

model.cpu()
plot_reconstructions(model, save=False, name='simple_bce', conv=True, simple=False)

'''Train Epoch: 300 [0/60000 (0%)]	Loss: 0.113431
Train Epoch: 300 [10000/60000 (17%)]	Loss: 0.113293
Train Epoch: 300 [20000/60000 (33%)]	Loss: 0.113812
Train Epoch: 300 [30000/60000 (50%)]	Loss: 0.112622
Train Epoch: 300 [40000/60000 (67%)]	Loss: 0.113216
Train Epoch: 300 [50000/60000 (83%)]	Loss: 0.112817'''