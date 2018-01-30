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
    def __init__(self, z_dim, hidden_dim, y_dim):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(z_dim + y_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 784)

        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        z = torch.cat(z, 1)
        hidden = self.softplus(self.fc1(z))
        img_mu = self.sigmoid(self.fc2(hidden))
        return img_mu


class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, y_dim):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(784 + y_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)

        self.softplus = nn.Softplus()

    def forward(self, x):
        x = torch.cat(x, 1)
        hidden = self.softplus(self.fc1(x))

        z_mu = self.fc21(hidden)
        z_sigma = torch.exp(self.fc22(hidden))
        return z_mu, z_sigma

class Encoder_y(nn.Module):
    def __init__(self, y_dim, hidden_dim):
        super(Encoder_y, self).__init__()

        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, y_dim)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax()

    def forward(self, x):
        hidden = self.softplus(self.fc1(x))
        return self.softmax(self.fc2(hidden))


class VAE(nn.Module):
    def __init__(self, z_dim=50, y_dim=10, hidden_dim=500, use_cuda=False):
        super(VAE, self).__init__()

        self.encoder = Encoder(z_dim, hidden_dim, y_dim)
        self.decoder = Decoder(z_dim, hidden_dim, y_dim)
        self.encoder_y = Encoder_y(y_dim, hidden_dim)

        if use_cuda:
            self.cuda()

        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.aux_loss_multiplier = 0.3

    def model(self, x, y=None):
        pyro.module('decoder', self.decoder)

        z_mu = ng_zeros([x.size(0), self.z_dim], type_as=x.data)
        z_sigma = ng_ones([x.size(0), self.z_dim], type_as=x.data)

        z = pyro.sample('latent', dist.normal, z_mu, z_sigma)

        alpha_prior = ng_ones([x.size(0), self.y_dim], type_as=y.data) / 10.
        if y is None:
            y = pyro.sample('y', dist.one_hot_categorical, alpha_prior)
        else:
            pyro.observe('y', dist.one_hot_categorical, y, alpha_prior)

        img_mu = self.decoder([z, y])
        pyro.observe('x', dist.bernoulli, x, img_mu)

    def guide(self, x, y=None):
        pyro.module('encoder', self.encoder)

        if y is None:
            alpha = self.encoder_y(x)
            y = pyro.sample('y', dist.one_hot_categorical, alpha)

        z_mu, z_sigma = self.encoder([x, y])
        z = pyro.sample("latent", dist.normal, z_mu, z_sigma)

    def model_classify(self, x, y):
        alpha = self.encoder_y(x)
        pyro.observe('y_aux', dist.one_hot_categorical,
                     y, alpha, log_pdf_mask=self.aux_loss_multiplier)

    def guide_classify(self, x, y):
        pass

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

    svi = SVI(vae.model, vae.guide, optimizer, loss='ELBO', enum_discrete=True)
    svi_aux = SVI(vae.model_classify, vae.guide_classify, optimizer, loss='ELBO')

    test_img = Variable(test_loader.dataset[0][0])

    yp = torch.FloatTensor(batch_size, 10)

    for epoch in range(epochs):
        epoch_loss = 0
        for x, y in train_loader:
            yp.zero_()
            yp.scatter_(1, y.view(-1, 1), 1)

            x = x.view(-1, 784)
            x, y = Variable(x), Variable(yp)

            loss = svi.step(x, y)
            loss = svi_aux.step(x,y)
            epoch_loss += loss
            print(loss)
        print(epoch_loss / len(train_loader.dataset))
        recon_img = vae.reconstruct_img(test_img)
        plt.gray()
        plt.subplot(1, 2, 1)
        plt.imshow(test_img.view(-1, 28, 28).squeeze(0).data.numpy())
        plt.subplot(1, 2, 2)
        plt.imshow(recon_img.view(-1, 28, 28).squeeze(0).data.numpy())
        plt.show()



