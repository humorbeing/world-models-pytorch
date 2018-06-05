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

transf = transforms.Compose([
                       transforms.ToTensor(),
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
    batch_size=10000,
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
    if not conv:
        data = data.view([-1, 784])
    data = Variable(data)#, volatile=True)
    true_imgs = data
    encoded_imgs = model.encoder(data)
    if simple:
        encoded_imgs = F.relu(encoded_imgs)
    decoded_imgs = model.decoder(encoded_imgs)

    true_imgs = to_img(true_imgs)
    decoded_imgs = to_img(decoded_imgs)

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
    def __init__(self, input_dim, encoding_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Linear(input_dim, encoding_dim)
        self.decoder = nn.Linear(encoding_dim, input_dim)

    def forward(self, x):
        encoded = F.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded, encoded


input_dim = 784
encoding_dim = 32

model = AutoEncoder(input_dim, encoding_dim)
model.cuda()
optimizer = optim.Adam(model.parameters())


def l1_penalty(var):
    return torch.abs(var).sum()


def train(epoch, sparsity=False, l1_weight=1e-5):
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data.view([-1, 784]).cuda())
        optimizer.zero_grad()

        # enforcing sparsity with l1 reg
        if sparsity:
            decoder_out, encoder_out = model(data)
            mse_loss = F.mse_loss(decoder_out, data)
            l1_reg = l1_weight * l1_penalty(encoder_out)
            loss = mse_loss + l1_reg
        else:
            output, _ = model(data)
            loss = F.binary_cross_entropy_with_logits(output, data)
            # loss = F.mse_loss(output, data)

        loss.backward()
        optimizer.step()
        # print(epoch)
        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                # loss.data[0]
                loss.item()
            ))


num_epochs = 30

for epoch in range(1,  num_epochs + 1):
    train(epoch)

model.cpu()
plot_reconstructions(model, save=False, name='simple_bce', simple=True)

'''Train Epoch: 300 [0/60000 (0%)]	Loss: 0.113431
Train Epoch: 300 [10000/60000 (17%)]	Loss: 0.113293
Train Epoch: 300 [20000/60000 (33%)]	Loss: 0.113812
Train Epoch: 300 [30000/60000 (50%)]	Loss: 0.112622
Train Epoch: 300 [40000/60000 (67%)]	Loss: 0.113216
Train Epoch: 300 [50000/60000 (83%)]	Loss: 0.112817'''