import os

import pandas as pd

import torch
import torch.nn
import torch.utils.data as data
from torch.autograd import Variable

import torchvision.transforms as transforms

class LastFMCSVDataset(data.Dataset):
    """Last.fm dataset."""

    def __init__(self, root_dir, domain, batch_rate, transform=None, separater='\t'):
        super(LastFMCSVDataset, self).__init__()
        source, target = domain

        csv_file = '{}_{}.dat'.format(source, target)
        csv_file = os.path.join(root_dir, csv_file)
        self.matrix = pd.read_csv(csv_file, separater)
        
        source_file = '{}.dat'.format(source)
        source_file = os.path.join(root_dir, source_file)
        self.source = pd.read_csv(source_file, separater)

        target_file = '{}.dat'.format(target)
        target_file = os.path.join(root_dir, target_file)
        self.target = pd.read_csv(target_file, separater)
        
        if isinstance(batch_rate, float):
            self.batch_size = int(len(self.matrix) * batch_rate)
        else:
            self.batch_size = batch_rate

        self.transform = transform
        self.domain = '{} x {} = {} x {}'.format(source, target, len(self.source), len(self.target))

    def __repr__(self):
        return self.domain

    def __len__(self):
        return len(self.matrix)

    def __getitem__(self, index):
        row, col, value = self.matrix.iloc[index]
        row_index = self.source[self.source.iloc[:, 0] == row].index
        col_index = self.target[self.target.iloc[:, 0] == col].index
        item = (row_index[0], col_index[0]), value

        if self.transform:
            item = self.transform(item)

        return item


class ToTensor(object):
    def __call__(self, item):
        (row, col), value = item
        row = torch.LongTensor([int(row)])
        col = torch.LongTensor([int(col)])
        value = torch.FloatTensor([float(value)])

        return (row, col), value


def setup_data_loaders(dataset, use_cuda, batch_rate, transform=False, root='./data/last.fm', **kwargs):
    if 'num_workers' not in kwargs:
        kwargs = {'num_workers': 0, 'pin_memory': False}

    cached_data = {}
    loaders = {}

    meta_domain = {'X': ('user', 'artists'),
                   'Y': ('tag', 'artists'),
                   'Z': ('user', 'user')}

    for kind in meta_domain.keys():
        cached_data[kind] = dataset(root, meta_domain[kind], batch_rate, transform=transform) 
        loaders[kind] = data.DataLoader(
            cached_data[kind], cached_data[kind].batch_size, shuffle=True, **kwargs)
    return loaders

class ToVariable(object):
    def __init__(self, use_cuda=False):
        self.use_cuda = use_cuda

    def __call__(self, item):
        (row, col), value = item
        row = Variable(row).cuda() if self.use_cuda else Variable(row)
        col = Variable(col).cuda() if self.use_cuda else Variable(col)
        value = Variable(value).cuda() if self.use_cuda else Variable(value)
        return (row, col), value

        
