import torch
import os
import networkx as nx
import numpy as np

from sklearn.metrics import average_precision_score
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

def random_shortcuts(data_list, th=1):
    """
    parameters:
        data_list[List]: 
            A list of PyG data objects has a length of n (n is the number of samples).

        th[float]: 
            A maximum threhold for the number of edges as a percentage of the number of edges of original graphs. 
                For example: 1 is adding 100% of the number of edges of original graph, 0.5 is adding 50%. Default is 1 or 100%. 
            
    return:
        new_data_list[List]:
            A list of processed PyG data objects with same length as input data_list.
    """
    new_data_list = []
    for data in data_list:
        edgelist = data.edge_index.T.numpy()
        new_edgelist = []

        while len(new_edgelist) <= len(edgelist)*th: 
            n = data.x.shape[0]
            a = np.random.choice(n)
            b = np.random.choice(n)
            if a == b:
                continue

            new_edgelist.append([a, b])

        new_edgelist = torch.LongTensor(np.concatenate([edgelist, new_edgelist]).T)
        
        new_data = Data(x=data.x, y=data.y, edge_index=new_edgelist, edge_attr=data.edge_attr)
        new_data_list.append(new_data)
        
    return new_data_list