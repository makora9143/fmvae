import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision.transforms import transforms

from dataset import setup_data_loaders, LastFMCSVDataset, ToTensor
import dataset
from model import FVAE, FMVAE

# def train_epoch(fmvae, optimizer, loaders, variable):
#     x_loader, y_loader, z_loader = iter(loaders['X']), iter(loaders['Y']), iter(loaders['Z'])
#
#     x_batches = len(loaders['X'])
#     y_batches = len(loaders['Y'])
#     z_batches = len(loaders['Z'])
#     min_batches = min(x_batches, y_batches, z_batches)
#     n = 36
#
#     for b in range(n):
#
#         losses = 0
#         for i in range(int(x_batches/n)):
#             data = variable(next(x_loader))
#             data_kind = 'X'
#             optimizer.zero_grad()
#             loss = - fmvae(data, data_kind)
#             loss.backward()
#             optimizer.step()
#             losses += loss.data[0]
#
#         for i in range(int(y_batches / n)):
#             data = variable(next(y_loader))
#             data_kind = 'Y'
#             optimizer.zero_grad()
#             loss = - fmvae(data, data_kind)
#             loss.backward()
#             optimizer.step()
#             losses += loss.data[0]
#
#         for i in range(int(z_batches / n)):
#             data_kind = 'Z'
#             data = variable(next(z_loader))
#             optimizer.zero_grad()
#             loss = - fmvae(data, data_kind)
#             loss.backward()
#             optimizer.step()
#             losses += loss.data[0]
#
#         print(losses)
#
def train_epoch(fmvae, optimizer, loaders, variable):
    x_loader, y_loader, z_loader = iter(loaders['X']), iter(loaders['Y']), iter(loaders['Z'])

    x_batches = len(loaders['X'])
    y_batches = len(loaders['Y'])
    z_batches = len(loaders['Z'])
    min_batches = min(x_batches, y_batches, z_batches)
    n = 36
    batch_per_epoch = x_batches + y_batches + z_batches

    x_ctr = 0
    y_ctr = 0
    z_ctr = 0

    for _ in range(n):
        optimizer.zero_grad()
        loss = 0
        for i in range(int(x_batches/n)):
            x_ctr += 1
            data = variable(next(x_loader))
            data_kind = 'X'
            optimizer.zero_grad()
            loss += - fmvae(data, data_kind)        


        for i in range(int(y_batches / n)):
            y_ctr += 1
            data = variable(next(y_loader))
            data_kind = 'Y'
            loss += - fmvae(data, data_kind)

        for i in range(int(z_batches / n)):
            z_ctr += 1
            data_kind = 'Z'
            data = variable(next(z_loader))
            loss += - fmvae(data, data_kind)

        loss.backward()
        optimizer.step()
        print(loss.data[0])

def train():
    compressed = transforms.Compose([dataset.ToTensor()])
    variable = dataset.ToVariable(use_cuda=True)
    kwargs = {'num_workers': 2, 'pin_memory': True}
    loaders = dataset.setup_data_loaders(dataset.LastFMCSVDataset, False, 32, transform=compressed, **kwargs)
    fmvae = FMVAE(1892, 17632, 11946, 1892, 1, 1000, 20)
    # fmvae = nn.DataParallel(fmvae)
    fmvae.cuda()

    adagrad_param = {'lr': 0.0001}
    optimizer = optim.Adam(fmvae.parameters(), **adagrad_param)

    epochs = 100
    for epoch in range(epochs):
        train_epoch(fmvae, optimizer, loaders, variable)



if __name__ == '__main__':
    train()
