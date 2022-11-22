import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BatchNorm1d, ReLU, Linear, Sequential
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, GINConv, global_add_pool, SAGEConv, to_hetero, HeteroConv, GraphConv
import os
from torch.utils.data import DataLoader
from torch_geometric.data import InMemoryDataset, download_url, HeteroData
import networkx as nx
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

data = HeteroData()
data['inst'].x = torch.FloatTensor(np.load("/home/zluo/nn/GNN/to_gnn/train_inst_X.npy"))
data['net'].x = torch.FloatTensor(np.load("/home/zluo/nn/GNN/to_gnn/train_net_X.npy").reshape(-1, 1))
data['net'].y = torch.FloatTensor(np.load("/home/zluo/nn/GNN/to_gnn/net_Y.npy").reshape(-1, 1))
data['inst', 'to', 'net'].edge_index  =  torch.LongTensor(np.load("/home/zluo/nn/GNN/to_gnn/edgs_inst_to_net.npy").T)
data['net', 'to', 'inst'].edge_index  =  torch.LongTensor(np.load("/home/zluo/nn/GNN/to_gnn/edgs_net_to_inst.npy").T)
data['inst', 'to', 'inst'].edge_index  = torch.LongTensor(np.load("/home/zluo/nn/GNN/to_gnn/edge_index_train_inst.npy").T)

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('inst', 'to', 'net'): GATConv((-1, -1), hidden_channels, add_self_loops=False, dropout=0.6),
                ('net', 'to', 'inst'): GATConv((-1, -1), hidden_channels, add_self_loops=False, dropout=0.6),
            }, aggr='sum')
            self.convs.append(conv)

        self.lin1 = Linear(hidden_channels, out_channels)
        self.lin2 = Linear(out_channels, 1)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
            
        x = F.relu(self.lin1(x_dict['net']))
        x = self.lin2(x)
        
        return x
    
    
model = HeteroGNN(hidden_channels=32, out_channels=10, num_layers=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()


with open("/home/zluo/nn/GNN/wl.log", "w", buffering=1) as f:
    for epoch in range(1, 10000):
        model.train()
        out = model(data.x_dict, data.edge_index_dict)
        loss = criterion(out, data['net'].y)
        loss.backward()
        optimizer.step()
        f.write(f'Epoch: {epoch:03d}, Loss: {loss:.4f}\n')
