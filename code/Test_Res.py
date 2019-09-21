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
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
from inputsdata import MyOwnDataset
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import SplineConv, voxel_grid, max_pool, max_pool_x, graclus, global_mean_pool, NNConv




transform = T.Cartesian(cat=False)
def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResidualBlock, self).__init__()
        self.left_conv1 = SplineConv(in_channel, out_channel, dim=2, kernel_size=5)
        self.left_bn1 = torch.nn.BatchNorm1d(out_channel)
        self.left_conv2 = SplineConv(out_channel, out_channel, dim=2, kernel_size=5)
        self.left_bn2 = torch.nn.BatchNorm1d(out_channel)
        
        self.shortcut_conv = SplineConv(in_channel, out_channel, dim=2, kernel_size=1)
        self.shortcut_bn = torch.nn.BatchNorm1d(out_channel)
        
     
    def forward(self, data):
        
        data.x = F.elu(self.left_bn2(self.left_conv2(F.elu(self.left_bn1(self.left_conv1(data.x, data.edge_index, data.edge_attr))),
                                            data.edge_index, data.edge_attr)) + 
                       self.shortcut_bn(self.shortcut_conv(data.x, data.edge_index, data.edge_attr)))
        
        return data
        
                    
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(1, 64, dim=2, kernel_size=5)
        self.bn1 = torch.nn.BatchNorm1d(64)  
        
        self.block1 = ResidualBlock(64, 128)
        self.block2 = ResidualBlock(128, 256)
        self.block3 = ResidualBlock(256, 512)

        self.fc1 = torch.nn.Linear(64 * 512, 1024)
        self.fc2 = torch.nn.Linear(1024, 10)
        
    def forward(self, data):
        data.x = F.elu(self.bn1(self.conv1(data.x, data.edge_index, data.edge_attr)))
        cluster = voxel_grid(data.pos, data.batch, size=[4,4])
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))
        
        data = self.block1(data)
        cluster = voxel_grid(data.pos, data.batch, size=[6,6])
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))
        
        data = self.block2(data)
        cluster = voxel_grid(data.pos, data.batch, size=[20,20])
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))
        
        data = self.block3(data)
        cluster = voxel_grid(data.pos, data.batch, size=[32,32])
        x = max_pool_x(cluster, data.x, data.batch, size=64)
        
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

#model = torch.load('./runs_model_res/model.pkl')
#model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_batch_logger = Logger(os.path.join('./Results', 'train_batch.log'), ['epoch', 'batch', 'loss', 'acc'])
test_batch_logger = Logger(os.path.join('./Results', 'test_batch.log'), ['batch', 'loss', 'acc'])
acc_logger = Logger(os.path.join('./Results', 'acc.log'), ['epoch', 'acc'])



#shutil.rmtree(osp.join('..',  'data/Traingraph/processed'))
#shutil.rmtree(osp.join('..', 'data/Testgraph/processed'))
for epoch in range(1, 150):
    
    train_path = osp.join('..',  'data/Traingraph')
    test_path = osp.join('..', 'data/Testgraph')

    train_data_aug = T.Compose([T.Cartesian(cat=False), T.RandomFlip(axis=0, p=0.5), T.RandomScale([0.96,0.999]), T.RandomFlip(axis=1, p=0.5)])
    test_data_aug = T.Compose([T.Cartesian(cat=False), T.RandomFlip(axis=0, p=0.5), T.RandomScale([0.96,0.999]), T.RandomFlip(axis=1, p=0.5)])

    train_dataset = MyOwnDataset(train_path, transform=train_data_aug)      #### transform=T.Cartesian()
    test_dataset = MyOwnDataset(test_path, transform=test_data_aug)

    
    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=3)
    
    train(epoch, train_batch_logger, train_loader)
    test_acc = test(test_batch_logger, test_loader)
    
    print('Epoch: {:02d}, Test: {:.4f}'.format(epoch, test_acc))
    acc_logger.log({'epoch': epoch, 'acc': test_acc})
    
    torch.save(model, './runs_model/model.pkl')
    
    
    shutil.rmtree(osp.join('..',  'data/Traingraph/processed'))
    #shutil.rmtree(osp.join('..', 'data/Testgraph/processed'))

    
    
