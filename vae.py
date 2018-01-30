import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
from pyro.optim import Adam
from pyro.infer import SVI
from pyro.util import ng_zeros, ng_ones

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt


class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 784)

        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        hidden = self.softplus(self.fc1(z))
        img_mu = self.sigmoid(self.fc2(hidden))
        return img_mu


class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)

        self.softplus = nn.Softplus()

    def forward(self, x):
        x = x.view(-1, 784)
        hidden = self.softplus(self.fc1(x))

        z_mu = self.fc21(hidden)
        z_sigma = torch.exp(self.fc22(hidden))
        return z_mu, z_sigma


class VAE(nn.Module):
    def __init__(self, z_dim=50, hidden_dim=400, use_cuda=False):
        super(VAE, self).__init__()

        self.encoder = Encoder(z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim)

        if use_cuda:
            self.cuda()

        self.use_cuda = use_cuda
        self.z_dim = z_dim

    def model(self, x):
        pyro.module('decoder', self.decoder)

        z_mu = ng_zeros([x.size(0), self.z_dim], type_as=x.data)
        z_sigma = ng_ones([x.size(0), self.z_dim], type_as=x.data)

        z = pyro.sample('latent', dist.normal, z_mu, z_sigma)

        img_mu = self.decoder(z)
        pyro.sample('obs', dist.bernoulli, img_mu, obs=x.view(-1, 784))

    def guide(self, x):
        pyro.module('encoder', self.encoder)

        z_mu, z_sigma = self.encoder(x)
        pyro.sample("latent", dist.normal, z_mu, z_sigma)

    def reconstruct_img(self, x):
        z_mu, z_sigma = self.encoder(x)
        z = dist.normal(z_mu, z_sigma)
        img_mu = self.decoder(z)
        return img_mu

    def generate(self, batch_size=1):
        prior_mu = Variable(torch.zeros([batch_size, self.z_dim]))
        prior_sigma = Variable(torch.ones([batch_size, self.z_dim]))

        zs = pyro.sample('z', dist.normal, prior_mu, prior_sigma)
        img_mu = self.decoder(zs)
        xs = pyro.sample('sample', dist.bernoulli, img_mu)
        return img_mu, xs


if __name__ == '__main__':
    batch_size = 100
    epochs = 10
    kwargs = {'num_workers': 1, 'pin_memory': True} if False else {}
    train_loader = data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    vae = VAE(use_cuda=False)
    
    adam_args = {'lr': 0.01}
    optimizer = Adam(adam_args)
    
    svi = SVI(vae.model, vae.guide, optimizer, loss='ELBO')

    test_img = Variable(test_loader.dataset[0][0])

    for epoch in range(epochs):
        epoch_loss = 0
        for x, y in train_loader:
            loss = svi.step(Variable(x))
            epoch_loss += loss
        print(epoch_loss / len(train_loader.dataset))
        recon_img = vae.reconstruct_img(test_img)
        plt.gray()
        plt.subplot(1, 2, 1)
        plt.imshow(test_img.view(-1, 28, 28).squeeze(0).data.numpy())
        plt.subplot(1, 2, 2)
        plt.imshow(recon_img.view(-1, 28, 28).squeeze(0).data.numpy())
        plt.show()


        
