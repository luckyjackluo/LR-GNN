#!/usr/bin/env python

import sys
import json


from src.data.test import load_dataset

def main(targets):
    if 'test' in targets:
        with open('config/test-params.json') as fh:
            test_params = json.load(fh)
        load_dataset(**test_params)

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
