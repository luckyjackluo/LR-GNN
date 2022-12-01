import os
import torch
import numpy as np
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from .peptides_functional import *

def load_dataset(save_path, n):

    print("Start testing loading the PyG dataset")

    dataset = PeptidesFunctionalDataset(save_path)
    idx_lst = np.random.choice([i for i in range(len(dataset))], size=n, replace=False)
    dataset = dataset[idx_lst]
    
    Y = []
    for data in dataset:
        Y.append(data.y.tolist()[0])

    new_data = [list(y) for y in set([tuple(x) for x in Y])]
    
    input_channels = dataset.num_node_features
    output_channels = len(new_data)
    
    one_hot = []
    for y in Y:
        to_add = [0 for idx in range(len(new_data))]
        to_add[new_data.index(y)] = 1
        one_hot.append(to_add)
        
    data_lst = []
    for idx in range(len(dataset)):
        data = dataset[idx]
        data.y = torch.FloatTensor([one_hot[idx]])
        data_lst.append(data)
    
    
    print("Dataset loaded and shuffled!")
    return data_lst, input_channels, output_channels


