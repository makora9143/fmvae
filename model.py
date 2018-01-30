import math

import torch
import torch.nn as nn
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.tmp = nn.Linear(hidden_dim, hidden_dim)
        # self.tmp2 = nn.Linear(hidden_dim, hidden_dim)
        # self.tmp3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        self.activation = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, x):
        hidden = self.activation(self.fc1(x))

        # hidden = self.softplus(self.tmp(hidden))
        # hidden = self.softplus(self.tmp2(hidden))
        # hidden = self.softplus(self.tmp3(hidden))

        z_mu = self.fc21(hidden)
        z_sigma = self.softplus(self.fc22(hidden))

        return z_mu, z_sigma

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, input_dim)
        self.fc22 = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, z):
        hidden = self.activation(self.fc1(z))

        x_mean = self.fc21(hidden)
        x_sigma = self.softplus(self.fc22(hidden))

        return x_mean, x_sigma


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim, kld=None):
        super(VAE, self).__init__()

        self.encoder = Encoder(input_dim, hidden_dim, z_dim)
        self.decoder = Decoder(input_dim, hidden_dim, z_dim)

        if kld is None:
            self.kld = self.kl_divergence
        else:
            self.kld = kld

    def forward(self, index, x):
        z_mu, z_sigma = self.encoder(x)
        z = self.rsampling(z_mu, z_sigma)
        recon_x_mu, recon_x_sigma = self.decoder(z)

        elbo = self.log_likelihood_normal(x, recon_x_mu, recon_x_sigma) - self.kld(z_mu, z_sigma)

        return torch.mean(elbo)

    def rsampling(self, mu, sigma):
        eps = Variable(torch.randn(mu.size()))
        return mu + sigma * eps

    def kl_divergence(self, mu, sigma):
        return - 0.5 * torch.sum(1 + torch.log(sigma) - torch.pow(mu, 2) - sigma, 1)

    def log_likelihood_normal(self, x, mu, sigma):
        return - 0.5 * torch.sum(torch.log(2 * math.pi * sigma) - torch.pow(x - mu, 2) / (2 * sigma), 1)


class FVAE(nn.Module):
    def __init__(self, row_num, col_num, input_dim, hidden_dim, z_dim):
        super(FVAE, self).__init__()
        
        self.U = nn.Embedding(row_num, z_dim, sparse=True)
        self.V = nn.Embedding(col_num, z_dim, sparse=True)

        self.encoder = Encoder(input_dim, hidden_dim, z_dim)
        self.decoder = Decoder(input_dim, hidden_dim, z_dim)

    def forward(self, index, x):
        z_mu, z_sigma = self.encoder(x)
        z = self.rsampling(z_mu, z_sigma)
        recon_x_mu, recon_x_sigma = self.decoder(z)

        ll = self.log_likelihood_normal(x, recon_x_mu, recon_x_sigma)
        d_kl = self.kl_divergence(z_mu, z_sigma, index)
        elbo = ll - d_kl
        return elbo

    def kl_divergence(self, mu, sigma, index):
        U_i = self.U(index[0]).squeeze(1)
        V_j = self.V(index[1]).squeeze(1)

        prior_mu = U_i * V_j
        prior_sigma = torch.ones(prior_mu.size())

        return 0.5 * torch.sum((torch.pow(prior_mu - mu, 2) + sigma - torch.log(sigma) - 1), 1)

    def rsampling(self, mu, sigma):
        eps = Variable(torch.randn(mu.size()))
        return mu + sigma * eps

    def log_likelihood_normal(self, x, mu, sigma):
        return torch.sum(- 0.5 * torch.log(2 * math.pi * sigma) - torch.pow(x - mu, 2) / (2 * sigma))


class FMVAE(nn.Module):
    def __init__(self, I, J, N, M, input_dim, hidden_dim, z_dim):
        super(FMVAE, self).__init__()
        self.W = nn.Embedding(I, z_dim, sparse=True)
        self.H = nn.Embedding(J, z_dim, sparse=True)
        self.A = nn.Embedding(N, z_dim, sparse=True)
        self.B = nn.Embedding(M, z_dim, sparse=True)

        self.x_encoder = Encoder(input_dim, hidden_dim, z_dim)
        self.x_decoder = Decoder(input_dim, hidden_dim, z_dim)

        self.y_encoder = Encoder(input_dim, hidden_dim, z_dim)
        self.y_decoder = Decoder(input_dim, hidden_dim, z_dim)

        self.z_encoder = Encoder(input_dim, hidden_dim, z_dim)
        self.z_decoder = Decoder(input_dim, hidden_dim, z_dim)


    def forward(self, data, data_kind):
        elbo = 0
        encoder = None
        decoder = None
        mat1 = None
        mat2 = None
        
        if data_kind == 'X':
            encoder = self.x_encoder
            decoder = self.x_decoder
            mat1 = self.W
            mat2 = self.H

        elif data_kind == 'Y':
            encoder = self.y_encoder
            decoder = self.y_decoder
            mat1 = self.A
            mat2 = self.H

        else:
            encoder = self.z_encoder
            decoder = self.z_decoder
            mat1 = self.W
            mat2 = self.B

        latent_mu, latent_sigma = encoder(data[1])
        latent = self.rsampling(latent_mu, latent_sigma)
        recon_mu, recon_sigma = decoder(latent)

        ll = torch.sum(self.log_likelihood_normal(data[1], recon_mu, recon_sigma))
        kld = torch.sum(self.kl_divergence(latent_mu, latent_sigma, data[0], mat1, mat2))
        return ll# - kld

    def kl_divergence(self, mu, sigma, index, U, V):
        U_i = U(index[0]).squeeze(1)
        V_j = V(index[1]).squeeze(1)

        prior_mu = U_i * V_j
        prior_sigma = torch.ones(prior_mu.size())

        return 0.5 * torch.sum((torch.pow(prior_mu - mu, 2) + sigma - torch.log(sigma) - 1), 1)

    def rsampling(self, mu, sigma):
        eps = Variable(torch.randn(mu.size()))
        return mu + sigma * eps

    def log_likelihood_normal(self, x, mu, sigma):
        # return - 0.5 * torch.sum(torch.log(2 * math.pi * sigma) - torch.pow(x - mu, 2) / (2 * sigma), 1)
        return torch.sum(- 0.5 * torch.log(2 * math.pi * sigma) - torch.pow(x - mu, 2) / (2 * sigma), 1)
