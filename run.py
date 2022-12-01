#!/usr/bin/env python

import sys
import json
import torch

from src.data.test import load_dataset
from src.models.models import *
from src.process.algorithms import random_shortcuts
from src.utils.utils import *

def main(targets):
    if 'test' in targets:
        with open('config/test-params.json') as fh:
            test_params = json.load(fh)
            
        data_lst, input_channels, output_channels = load_dataset(**test_params)
        model = GCN(input_channels, 16, output_channels)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        print("testing the baseline model's performances")
        print("start running model: GCN")
        model = GCN(input_channels, 16, output_channels)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        
        run(model, data_lst, 0.8, criterion, optimizer, 10, 10)
        
        
        
        print("using random shortcuts adding algorithm")
        new_data_lst = random_shortcuts(data_lst)
        print("shortcuts added, new data objects created")
        print("testing the new model's performances")
        print("start running model: GCN")
        model = GCN(input_channels, 16, output_channels)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        
        run(model, data_lst, 0.8, criterion, optimizer, 10, 10)


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
