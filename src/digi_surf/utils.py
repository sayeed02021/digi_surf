import yaml 
import argparse
import pickle


def load_yaml(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    return argparse.Namespace(**data)

def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    return data
