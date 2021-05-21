import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
import torchvision
import cv2
import numpy as np
from os.path import join, basename, dirname, exists
import json
from utils import get_paths
import pandas as pd

class SETIDataset(Dataset):
    
    def __init__(self, data_file_paths, targets, transform=None):
        """
        Initializes SETI dataset class.

        Parameters
        ----------
        data_folder : PATH-STR
            Path to parent data folder.
        labels_path : PATH-STR
            Path to label file.
        transform : FUNCTION, optional
            Function to preprocess a given cadence. The default is None.

        Returns
        -------
        None.

        """
        
        self.transform = transform
        self.data_file_paths = data_file_paths
        self.targets = targets
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        
        # Read file at given index
        data = np.load(self.data_file_paths[idx])
        data = data.astype(np.float32)
        
        # Is this a good idea?
        data = np.vstack(data).transpose((1, 0))
        
        # Perform augmentations if desired
        if not self.transform is None:
            data = self.transform(data)
        else:
            data = data[np.newaxis, :, :]
            data = torch.from_numpy(data).float()
        
        # Grab label, return
        label = torch.tensor(self.targets[idx]).float()
        return data, label
        