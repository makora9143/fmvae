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


class MF(nn.Module):
    def __init__(self, row_dim, col_dim, K):
        super(MF, self).__init__()
        self.U_size = (row_dim, K)
        self.V_size = (col_dim, K)

    def model(self, x):
        u_mu = ng_zeros(self.U_size)
        u_sigma = ng_ones(self.U_size)

        U = pyro.sample('u', dist.normal, u_mu, u_sigma)

        v_mu = ng_zeros(self.V_size)
        v_sigma = ng_ones(self.V_size)

        V = pyro.sample('v', dist.normal, v_mu, v_sigma)
        pyro.observe('x', dist.bernoulli,
                     x,
                     torch.matmul(U, torch.t(V)),
                     # ng_ones(x.size(), type_as=x.data)
                     )

    def guide(self, x):
        qu_mu = Variable(torch.randn(self.U_size), requires_grad=True)
        qu_sigma = Variable(torch.randn(self.U_size), requires_grad=True)

        qu_mu = pyro.param('qu_mu', qu_mu) 
        qu_sigma = pyro.param('qu_sigma', qu_sigma) 

        qU = pyro.sample('u', dist.normal, qu_mu, torch.exp(qu_sigma))

        qv_mu = Variable(torch.randn(self.V_size), requires_grad=True)
        qv_sigma = Variable(torch.randn(self.V_size), requires_grad=True)

        qv_mu = pyro.param('qv_mu', qv_mu)
        qv_sigma = pyro.param('qv_sigma', qv_sigma)

        qV = pyro.sample('v', dist.normal, qv_mu, torch.exp(qv_sigma))

    def forward(self):
        pass


if __name__ == '__main__':
    # R = Variable(torch.FloatTensor([
    #             [5, 3, 0, 1],
    #             [4, 0, 0, 1],
    #             [1, 1, 0, 5],
    #             [1, 0, 0, 4],
    #             [0, 1, 5, 4],
    #             ]
    #         ))
    from PIL import Image
    from PIL import ImageOps
    import torchvision.transforms as transforms
    img = Image.open('./Buson_Nopperabo.jpg')
    img = ImageOps.grayscale(img)
    img = transforms.ToTensor()(img)
    R = Variable(img.squeeze(0))

    mf = MF(R.size(0), R.size(1), 50)
    adam_params = {"lr": 0.0005}
    optimizer = Adam(adam_params)

    svi = SVI(mf.model, mf.guide, optimizer, loss='ELBO', num_particles=5)
    losses = []
    for epoch in range(10000):
        loss = svi.step(R)
        losses.append(loss)
        if epoch % 100 == 0:
            print(epoch, loss)
    
    import matplotlib.pyplot as plt

    plt.subplot(131)
    plt.plot(list(range(len(losses))), losses)
    plt.title("ELBO")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.subplot(132)
    plt.gray()
    plt.imshow(R.data.numpy())
    plt.subplot(133)
    qu = pyro.param('qu_mu')
    qv = pyro.param('qv_mu')
    recon = torch.matmul(qu, torch.t(qv)).data.numpy()
    plt.imshow(recon)
    plt.show()

    
