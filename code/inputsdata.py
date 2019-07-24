import collections
import os.path as osp
import os
import errno
import numpy as np
import glob
import scipy.io as sio
import torch

import torch.utils.data
from torch_geometric.data import Data, DataLoader, Dataset


import os.path as osp

import torch
from torch_geometric.data import Dataset


def files_exist(files):
    return all([osp.exists(f) for f in files])

class MyOwnDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        filenames = glob.glob(os.path.join(self.raw_dir, '*.mat'))
        #print(filenames)
        file = [f.split('/')[4] for f in filenames]
        #print(file)
        return file

    @property
    def processed_file_names(self):
        filenames = glob.glob(os.path.join(self.raw_dir, '*.mat'))
        file = [f.split('/')[4] for f in filenames]
        saved_file = [f.replace('.mat','.pt') for f in file]
        return saved_file

    def __len__(self):
        return len(self.processed_file_names)
    
    def download(self):
        if files_exist(self.raw_paths):
            return
        print('No found data!!!!!!!')


    def process(self):
        for raw_path in self.raw_paths:
             # Read data from `raw_path`.
            content = sio.loadmat(raw_path)
            feature = torch.tensor(content['feature'])
            edge_index = torch.tensor(np.array(content['edge'], np.int32), dtype=torch.long)
            pos = torch.tensor(content['pseudo'])

            label_idx = torch.tensor(content['label'], dtype=torch.long)
            #print(label_idx.shape)
                   
            data = Data(x=feature, edge_index=edge_index, pos=pos, y=label_idx.squeeze(0))

            if self.pre_filter is not None and not self.pre_filter(data):
                 continue

            if self.pre_transform is not None:
                 data = self.pre_transform(data)

            saved_name = raw_path.split('/')[4].replace('.mat','.pt')
            torch.save(data, osp.join(self.processed_dir, saved_name))

    def get(self, idx):
        data = torch.load(osp.join(self.processed_paths[idx]))
        return data




    
if __name__ == '__main__':


    root = os.path.join('..', 'subdata/Traingraph')
    MNIST = MyOwnDataset(root)
    print(MNIST[0])

    loader = DataLoader(MNIST, batch_size=3, shuffle=True)

    for data in loader:
        print(data)
        #x, edge_index, pseudo = data.x, data.edge_index, data.pos
        #out = SplineConv(1, 16, 3, 5)(x, edge_index, pseudo)