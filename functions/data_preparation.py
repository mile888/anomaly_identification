#!/usr/bin/env python

### Minimum Redundancy Maximum Relevance for classification ###

import numpy as np
import pandas as pd
import json

from sklearn.model_selection import train_test_split


def load_dataset(filename):
    df = pd.read_csv(filename)
    X = df.iloc[:, :-1] 
    y = df.iloc[:,-1]
    return df, X, y

def split_data(X, y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y) 
    return X_train, X_test, y_train, y_test

def to_dict(time_LR, time_KNN, time_RF, time_XGB):
       
    time_dict = {}
    time_dict['LR'] = time_LR
    time_dict['KNN'] = time_KNN
    time_dict['RF'] = time_RF
    time_dict['XGB'] = time_XGB

    return time_dict    
    
#Save and Load
def save_model(time_dict, filename):
        """ Save model to a json file"""
        with open(filename + ".json", "w") as outfile:
            json.dump(time_dict, outfile)
    
def load_model(filename):
        """ Create a new model from file"""
        with open(filename + ".json") as json_data:
            input_dict = json.load(json_data)
        return input_dict





