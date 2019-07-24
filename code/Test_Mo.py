from __future__ import division
import os.path as osp
import numpy as np
import math
import shutil
import os
from utils import Logger

import torch
from torch.utils.data.dataset import Subset
import torch.nn as nn 
from torch.nn import Sequential, Linear, ReLU
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
from inputsdata import MyOwnDataset
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import voxel_grid, max_pool, max_pool_x, graclus, global_mean_pool, GMMConv, NNConv, global_mean_pool




transform = T.Cartesian(cat=False)
def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = GMMConv(1, 64, dim=2)
        self.bn1 = torch.nn.BatchNorm1d(64)
        
        self.conv2 = GMMConv(64, 128, dim=2)
        self.bn2 = torch.nn.BatchNorm1d(128)
        
        self.conv3 = GMMConv(128, 256, dim=2)
        self.bn3 = torch.nn.BatchNorm1d(256)
        
        self.conv4 = GMMConv(256, 512, dim=2)
        self.bn4 = torch.nn.BatchNorm1d(512)
        
        self.fc1 = torch.nn.Linear(32*512, 1024)
        self.fc2 = torch.nn.Linear(1024, 10)

        


    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        data.x = self.bn1(data.x)
        cluster = voxel_grid(data.pos, data.batch, size=[4,4])
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        data.x = self.bn2(data.x)
        cluster = voxel_grid(data.pos, data.batch, size=[6,6])
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))
        
        data.x = F.elu(self.conv3(data.x, data.edge_index, data.edge_attr))
        data.x = self.bn3(data.x)
        cluster = voxel_grid(data.pos, data.batch, size=[20,20])
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))
        
        data.x = F.elu(self.conv4(data.x, data.edge_index, data.edge_attr))
        data.x = self.bn4(data.x)
        cluster = voxel_grid(data.pos, data.batch, size=[32,32])
        x = max_pool_x(cluster, data.x, data.batch, size=32)
        
        x = x.view(-1, self.fc1.weight.size(1))

        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


def train(epoch, batch_logger, train_loader):
    model.train()

    if epoch == 60:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    if epoch == 110:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.00001

    for i, data in enumerate(train_loader):
        with autograd.detect_anomaly():
            data = data.to(device)
            #print(data.y)
            optimizer.zero_grad()
            end_point = model(data)
            
            loss = F.nll_loss(end_point, data.y)
            pred = end_point.max(1)[1]
            acc = (pred.eq(data.y).sum().item())/len(data.y)
            
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                batch_logger.log({'epoch': epoch,'batch': i + 1,'loss': loss.item(),'acc': acc})
            



def test(batch_logger, test_loader):
    model.eval()
    correct = 0

    for i, data in enumerate(test_loader):
        data = data.to(device)
        end_point = model(data)
        loss = F.nll_loss(end_point, data.y)
        
        pred = end_point.max(1)[1]
        acc = (pred.eq(data.y).sum().item())/len(data.y)
        correct += acc
        
        if i % 10 == 0:
            batch_logger.log({'batch': i + 1,'loss': loss.item(),'acc': acc})
        

    return correct / (i+1)




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)

#model = torch.load('./runs_model/model.pkl')
#model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_batch_logger = Logger(os.path.join('./Results', 'train_batch.log'), ['epoch', 'batch', 'loss', 'acc'])
test_batch_logger = Logger(os.path.join('./Results', 'test_batch.log'), ['batch', 'loss', 'acc'])
acc_logger = Logger(os.path.join('./Results', 'acc.log'), ['epoch', 'acc'])


#shutil.rmtree(osp.join('..',  'data/Traingraph/processed'))
#shutil.rmtree(osp.join('..', 'data/Testgraph/processed'))
for epoch in range(1, 180):
    
    train_path = osp.join('..',  'data/Traingraph')
    test_path = osp.join('..', 'data/Testgraph')

    #train_data_aug = T.Compose([T.Cartesian(cat=False), T.RandomFlip(axis=0)])
    train_data_aug = T.Compose([T.Cartesian(cat=False), T.RandomFlip(axis=0, p=0.3), T.RandomScale([0.96,0.999]) ])
    test_data_aug = T.Compose([T.Cartesian(cat=False), T.RandomFlip(axis=0, p=0.3), T.RandomScale([0.96,0.999])])

    train_dataset = MyOwnDataset(train_path, transform=train_data_aug)      #### transform=T.Cartesian()
    test_dataset = MyOwnDataset(test_path, transform=test_data_aug)

    
    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=3)
    
    train(epoch, train_batch_logger, train_loader)
    test_acc = test(test_batch_logger, test_loader)
    
    print('Epoch: {:02d}, Test: {:.4f}'.format(epoch, test_acc))
    
    torch.save(model, './runs_model/model.pkl')
    acc_logger.log({'epoch': epoch, 'acc': test_acc})
    
    
    shutil.rmtree(osp.join('..',  'data/Traingraph/processed'))
    #shutil.rmtree(osp.join('..', 'data/Testgraph/processed'))

    
    
