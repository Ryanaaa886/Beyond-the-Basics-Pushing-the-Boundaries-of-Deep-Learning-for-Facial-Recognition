"""
Function: load_dataset

Load images from LFW datasets using sklearn library.
"""


import os
from sklearn.datasets import fetch_lfw_people, fetch_lfw_pairs


def load_dataset(data_home:str = "./lfw_dataset/", pairs:bool=True):
    """
    Load LFW dataset from sklearn

    Args:
        data_home: Specify another download and cache folder for the datasets. 
        pairs: Whether to get pairs data.
    """
    if not os.path.exists(data_home):
        os.mkdir(data_home)
    
    if pairs:
        train_pairs = fetch_lfw_pairs(subset="train", data_home=data_home, color=True)
        test_pairs = fetch_lfw_pairs(subset="test", data_home=data_home, color=True)
        
        return train_pairs, test_pairs
    else:
        dataset = fetch_lfw_people(data_home=data_home, color=True)
        
        return dataset
    

