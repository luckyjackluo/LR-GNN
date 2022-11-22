import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, Sequential
import os
from torch import nn
from torch_geometric.data import InMemoryDataset, download_url
import networkx as nx
from sklearn.preprocessing import StandardScaler

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.fc1 = nn.Linear(hidden_channels, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x, edge_index):
        #x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        #x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    
    
class netlist_wl(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['netlist_wl.gpickle']

    @property
    def processed_file_names(self):
        return ['netlist_wl.pt']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = []

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        X = np.load("/home/zluo/nn/GNN/graph_X.npy", allow_pickle=True)
        X = StandardScaler().fit_transform(X)
        y = np.load("/home/zluo/nn/GNN/graph_y.npy", allow_pickle=True)
        y = y.reshape(len(y), 1)
        edge_index = np.load("/home/zluo/nn/GNN/edge_index.npy")
        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        edge_index = torch.tensor(edge_index.T, dtype=torch.long)
        gp_data = Data(x=X, y=y, edge_index=edge_index)
        data_list.append(gp_data)
            
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        
        
dataset = netlist_wl("/home/zluo/nn/gnn/")
data = dataset[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load("/home/zluo/nn/GNN/trained.pt")
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = torch.nn.MSELoss()


def train():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out, data.y)  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss

with open("/home/zluo/nn/GNN/wl.log", "w", buffering=1) as f:
    for epoch in range(1, 10000):
        loss = train()
        f.write(f'Epoch: {epoch:03d}, Loss: {loss:.4f}\n')