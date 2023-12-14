import pickle
import gzip
import numpy as np
import os

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object
    
def save_zipped_pickle(obj, filename):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, 2)

if "__name__" == "__main__":
    # load data
    train_data = load_zipped_pickle("train.pkl")
    test_data = load_zipped_pickle("test.pkl")
    samples = load_zipped_pickle("sample.pkl")