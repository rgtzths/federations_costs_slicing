#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Rafael Teixeira'
__version__ = '0.1'
__email__ = 'rafaelgteixeira@ua.pt'
__status__ = 'Development'

import argparse
import pathlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

'''
Used to create the training, valdiation and test datasets.
Transforms and shuffles the dataset into the different subsets ready to be inserted into the model.
'''

def one_hot_encoding(input_file, output_folder, seed):
    df = pd.read_excel(input_file, sheet_name="Model_Inputs_Outputs")

    le = LabelEncoder()
    enc = OneHotEncoder(handle_unknown='error')

    output = pathlib.Path(output_folder)
    output.mkdir(parents=True, exist_ok=True)

    del df["Unnamed: 0"]
    df["LTE/5G UE Category (Input 2)"] = df["LTE/5G UE Category (Input 2)"].astype(str)
    df["Slice Type (Output)"] = le.fit_transform(df["Slice Type (Output)"])
    print(le.classes_)
    #Transform features into one hot vectors
    encoded_features = df.drop("Time (Input 5)", axis=1)
    encoded_features = encoded_features.drop("Slice Type (Output)", axis=1)
    
    enc.fit(encoded_features)
    data = enc.transform(encoded_features).toarray()
    data = np.append(data, df["Time (Input 5)"].to_numpy().reshape(-1,1), axis=1)
    data = np.append(data, df["Slice Type (Output)"].to_numpy().reshape(-1,1), axis=1)

    X = data[:,:-1]
    y = data[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state= seed, test_size=0.2)

    X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, random_state=seed, test_size=y_test.shape[0])
    
    np.savetxt(output/"x_train.csv", X_train, delimiter=",", fmt="%d")
    np.savetxt(output/"x_cv.csv", X_cv, delimiter=",", fmt="%d")
    np.savetxt(output/"x_test.csv", X_test, delimiter=",", fmt="%d")
    np.savetxt(output/"y_train.csv", y_train, delimiter=",", fmt="%d")
    np.savetxt(output/"y_cv.csv", y_cv, delimiter=",", fmt="%d")
    np.savetxt(output/"y_test.csv", y_test, delimiter=",", fmt="%d")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate train/validation/test data from the raw data')
    parser.add_argument('-f', type=str, help='Dataset file', default='raw_data/data.xlsx')
    parser.add_argument('-o', type=str, help='Output folder', default='processed_data/')
    parser.add_argument('-s', type=int, help='Seed used for data shuffle', default=3)
    args = parser.parse_args()

    one_hot_encoding(args.f, args.o, args.s)