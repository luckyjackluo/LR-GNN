import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_add_pool, GATConv
import os
from torch import nn
from torch_geometric.data import InMemoryDataset, download_url
import networkx as nx
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

dataset = "enzyme"

if dataset == "netlist":
    class netlist(InMemoryDataset):
        def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
            super().__init__(root, transform, pre_transform, pre_filter)
            self.data, self.slices = torch.load(self.processed_paths[0])

        @property
        def raw_file_names(self):
            return ['netlist.gpickle']

        @property
        def processed_file_names(self):
            return ['netlist.pt']

        def download(self):
            pass

        def process(self):
            # Read data into huge `Data` list.
            data_list = []

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            paths = np.load("/home/zluo/new_chip_design/paths.npy")

            path_lst = []
            for f in paths:
                path_lst.append(f.split("_")[:-1])

            path_lst = np.array([lst[1] for lst in path_lst]).reshape(-1, 1)
            encoder = OneHotEncoder()
            labels = encoder.fit_transform(path_lst).toarray()
            labels = torch.tensor(labels, dtype=torch.float)

            X_eig = np.load("/home/zluo/new_chip_design/eig_all.npy")
            X_new = np.load("/home/zluo/new_chip_design/new_X.npy")

            X_eig = StandardScaler().fit_transform(X_eig)
            X_new = StandardScaler().fit_transform(X_new)

            for index in range(len(paths)):
                path = paths[index]
                netlist_name = path.split(".")[0]
                g_path = f"/home/zluo/new_chip_design/gps/{netlist_name}.gpickle"
                graph = nx.read_gpickle(g_path)
                nodelist = list(graph.nodes())
                nodedict = {nodelist[idx]:idx for idx in range(len(nodelist))}
                X = torch.tensor([[graph.in_degree()[node], graph.out_degree()[node]] for node in nodelist], dtype=torch.float)
                y = labels[index]
                edge_index = torch.tensor([[nodedict[tp[0]], nodedict[tp[1]]] for tp in list(graph.edges())], dtype=torch.long).T
                gp_data = Data(x=X, y=y, edge_index=edge_index, eig=torch.tensor(X_eig[index], dtype=torch.float), stats=torch.tensor(X_new[index], dtype=torch.float))

                data_list.append(gp_data)

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])


    class GAT(torch.nn.Module):
        def __init__(self):
            super(GAT, self).__init__()
            self.hid = 8
            self.in_head = 8
            self.out_head = 1

            self.conv1 = GATConv(dataset.num_features, self.hid, heads=self.in_head, dropout=0.6)
            self.conv2 = GATConv(self.hid*self.in_head, self.hid, concat=False,
                                 heads=self.out_head, dropout=0.6)

            self.fc1 = nn.Linear(self.hid+10, 20)
            self.fc2 = nn.Linear(20, 5)

        def forward(self, x, edge_index, batch, eig, stats):

            # Dropout before the GAT layer is used to avoid overfitting in small datasets like Cora.
            # One can skip them if the dataset is sufficiently large.

            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv2(x, edge_index)

            x = global_mean_pool(x, batch)

            x = torch.cat((x, stats.reshape(x.shape[0], 10)), dim=1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)

            return F.log_softmax(x, dim=1)


    dataset = netlist("/home/zluo/nn/gnn/")
    #dataset.process()
    train_idx, test_idx = np.load("/home/zluo/new_chip_design/idx_lst.npy", allow_pickle=True)
    train_dataset = dataset[train_idx]
    test_dataset = dataset[test_idx]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GAT().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    with open("/home/zluo/nn/GNN/gnn.log", "w", buffering=1) as f:
        for epoch in range(10000):
            model.train()

            loss_train = 0
            loss_test = 0
            correct_train = 0
            correct_test = 0

            for data in train_dataset:
                data = data.to(device)
                optimizer.zero_grad()
                output = model(data.x, data.edge_index, data.batch, data.eig, data.stats)[0]
                pred = output.argmax(dim=0)  
                label = data.y.argmax(dim=0)  
                correct_train += int(pred==label)
                loss = F.cross_entropy(output, data.y)
                loss.backward()
                loss_train += loss.item()
                optimizer.step()

            for data in test_dataset:
                data = data.to(device)
                output = model(data.x, data.edge_index, data.batch, data.eig, data.stats)[0]
                pred = output.argmax(dim=0)  
                label = data.y.argmax(dim=0)  
                correct_test += int(pred==label)
                loss_t = F.cross_entropy(output, data.y)
                loss_test += loss_t.item()


            loss_train = loss_train/len(train_dataset)
            loss_test = loss_test/len(test_dataset)
            acc_train = correct_train/len(train_dataset)
            acc_test = correct_test/len(test_dataset)

            f.write(f"{epoch}, {loss_train}, {loss_test}, {acc_train}, {acc_test}\n")

            if epoch//1000 > 0 and epoch%1000 == 0:
                torch.save(model, "trained.pt")


else:
    
