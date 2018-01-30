import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision.transforms import transforms

from dataset import setup_data_loaders, LastFMCSVDataset, ToTensor
import dataset
from model import VAE

def train_epoch(vae, optimizer, loaders, variable):
    epoch_loss = 0
    for step, x in enumerate(loaders['X']):
        optimizer.zero_grad()
        x = variable(x)[1]
        loss = vae(x)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.data[0]
        if step % 100 == 0:
            print('{} step loss:{}'.format(step, loss.data[0]))
    return epoch_loss

def train():
    use_cuda = True
    epochs = 100
    batch_size = 32
    hidden_dim = 500
    z_dim = 20
    lr = 0.0001

    compressed = transforms.Compose([dataset.ToTensor()])
    variable = dataset.ToVariable(use_cuda=use_cuda)
    kwargs = {'num_workers': 2, 'pin_memory': True}
    loaders = dataset.setup_data_loaders(dataset.LastFMCSVDataset, use_cuda, batch_size, transform=compressed, **kwargs)

    print('{} steps for all data / 1 epoch'.format(len(loaders['X'])))

    vae = VAE(1, hidden_dim, z_dim, use_cuda=use_cuda)

    adagrad_param = {'lr': lr}
    optimizer = optim.Adam(vae.parameters(), **adagrad_param)

    for epoch in range(epochs):
        loss = train_epoch(vae, optimizer, loaders, variable)
        print('Epoch{}:{}'.format(epoch, loss))



if __name__ == '__main__':
    train()
