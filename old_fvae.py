from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data

import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST

import pyro
import pyro.distributions as dist
from pyro.util import ng_zeros, ng_ones
from pyro.optim import Adam
from pyro.infer import SVI

class MNISTIndexed(MNIST):
    def __init__(self,*args, **kwargs):
        super(MNISTIndexed, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = super(MNISTIndexed, self).__getitem__(index)

        return img, (index, target)

class FashionMNISTIndexed(FashionMNIST):
    def __init__(self,*args, **kwargs):
        super(FashionMNISTIndexed, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = super(FashionMNISTIndexed, self).__getitem__(index)

        return img, (index, target)

class Encoder(nn.Module):
    def __init__(self, z_dim, input_dim=784, hidden_dim=500):
        super(Encoder, self).__init__()

        self.input_dim = input_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)

        self.softplus = nn.Softplus()

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        hidden = self.softplus(self.fc1(x))
        z_mu = self.fc21(hidden)
        z_sigma2 = self.softplus(self.fc22(hidden))
        return z_mu, z_sigma2


class Decoder(nn.Module):
    def __init__(self, z_dim, input_dim=784, hidden_dim=500):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, input_dim)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        hidden = self.softplus(self.fc1(z))
        x_mu = self.sigmoid(self.fc21(hidden))
        return x_mu


class FVAE(nn.Module):

    def __init__(self, z_dim, u_dim, v_dim, input_dim=784, hidden_dim=500, use_cuda=False):
        super(FVAE, self).__init__()

        self.encoder = Encoder(z_dim, input_dim, hidden_dim)
        self.decoder = Decoder(z_dim, input_dim, hidden_dim)
        
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.v_dim = v_dim

        if use_cuda:
            self.cuda()

    def model(self, x, y):
        pyro.module('decoder', self.decoder)

        i, y = y

        U_mu = ng_zeros([self.u_dim, self.z_dim])
        U_sigma = ng_ones([self.u_dim, self.z_dim])
        U = pyro.sample('U', dist.normal, U_mu, U_sigma)

        V_mu = ng_zeros([self.v_dim, self.z_dim])
        V_sigma = ng_ones([self.v_dim, self.z_dim])
        V = pyro.sample('V', dist.normal, V_mu, V_sigma)

        z_sigma = ng_ones([x.size(0), self.z_dim], type_as=x.data)

        z = pyro.sample('latent', dist.normal, U[i, :] * V[y, :], z_sigma)

        img_mu = self.decoder(z)
        pyro.sample('obs', dist.bernoulli, img_mu, obs=x.view(-1, 784))

    def guide(self, x, y):
        pyro.module('encoder', self.encoder)

        qu_mu = Variable(torch.randn(self.u_dim, self.z_dim), requires_grad=True)
        qu_sigma = Variable(torch.randn(self.u_dim, self.z_dim), requires_grad=True)

        qu_mu = pyro.param('qu_mu', qu_mu) 
        qu_sigma = pyro.param('qu_sigma', qu_sigma) 

        qU = pyro.sample('U', dist.normal, qu_mu, torch.exp(qu_sigma))

        qv_mu = Variable(torch.randn(self.v_dim, self.z_dim), requires_grad=True)
        qv_sigma = Variable(torch.randn(self.v_dim, self.z_dim), requires_grad=True)

        qv_mu = pyro.param('qv_mu', qv_mu)
        qv_sigma = pyro.param('qv_sigma', qv_sigma)

        qV = pyro.sample('V', dist.normal, qv_mu, torch.exp(qv_sigma))

        z_mu, z_sigma = self.encoder(x)
        pyro.sample("latent", dist.normal, z_mu, z_sigma)


if __name__ == '__main__':
    batch_size = 100
    epochs = 10
    kwargs = {'num_workers': 1, 'pin_memory': True} if False else {}
    train_loader = data.DataLoader(
        FashionMNISTIndexed('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = data.DataLoader(
        FashionMNISTIndexed('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    vae = FVAE(6, 60000, 10, use_cuda=False)
    
    adam_args = {'lr': 0.01}
    optimizer = Adam(adam_args)
    
    svi = SVI(vae.model, vae.guide, optimizer, loss='ELBO')
    losses = []
    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            x = Variable(x.float().view(-1, 784))
            y = Variable(y[0]), Variable(y[1])
            loss = svi.step(x, y)
            print(loss)
            losses.append(loss)
        for i, (x, y) in enumerate(test_loader):
            x = Variable(x.float().view(-1, 784))
            y = Variable(y[0]), Variable(y[1])
            loss = svi.evaluate_loss(x, y)
            print(loss)
    import matplotlib.pyplot as plt

    plt.plot(list(range(len(losses))), losses)
    plt.show()

