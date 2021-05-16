import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
import torchvision
import cv2
import numpy as np
from os.path import join, basename, dirname, exists
import json
import pickle
from utils import get_paths

class SETIDataset(Dataset):
    
    def __init__(self, data_folder, labels, transform=None):
        """
        Initializes SETI dataset class.

        Parameters
        ----------
        data_folder : PATH-STR
            Path to parent data folder.
        labels : PATH-STR
            Path to label file.
        transform : FUNCTION, optional
            Function to preprocess a given cadence. The default is None.

        Returns
        -------
        None.

        """
        
        # Check for cached file names
        cache_fn = join(data_folder, "file_paths.cache")
        if exists(cache_fn):
            with open(cache_fn, 'rb') as fp:
                data_file_paths = pickle.load(fp)
        else:
            
            # Grab all relevant file paths
            data_file_paths = get_paths(data_folder, ext='npy')
            