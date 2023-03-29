#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Rafael Teixeira'
__version__ = '0.1'
__email__ = 'rafaelgteixeira@ua.pt'
__status__ = 'Development'

import argparse
import math
import pathlib

import numpy as np


def main(args):

    x_train = np.loadtxt(args.x, delimiter=",", dtype=int)
    y_train = np.loadtxt(args.y, delimiter=",", dtype=int)

    output = pathlib.Path(args.o)
    output.mkdir(parents=True, exist_ok=True)

    subset_size = math.ceil(len(x_train) / 3)
    subset = 1
    for i in range(0, len(x_train), subset_size):
        np.savetxt(output/f"x_train_subset_{subset}.csv", x_train[i:i+subset_size], delimiter=",", fmt="%d")
        np.savetxt(output/f"y_train_subset_{subset}.csv", y_train[i:i+subset_size], delimiter=",", fmt="%d")
        subset += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate equally sized subsets from two valid CSVs.')
    parser.add_argument('-x', type=str, help='X_train file', default='one_hot_encoding/x_train.csv')
    parser.add_argument('-y', type=str, help='y_train file', default='one_hot_encoding/y_train.csv')
    parser.add_argument('-o', type=str, help='Output folder', default='one_hot/')
    parser.add_argument('-n', type=int, help='Number of subsets', default=3)
    
    args = parser.parse_args()

    main(args)