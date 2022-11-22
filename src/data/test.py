import os
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

def load_dataset(save_path, name):

    dataset = TUDataset(root=save_path, name=name)
    dataset = dataset.shuffle()
    train_dataset = dataset[:150]
    test_dataset = dataset[150:]

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader


