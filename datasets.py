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
    
    def __init__(self, data_folder, labels_path, transform=None):
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
        
        # Check for cached file names
        cache_fn = join(data_folder, "file_paths.json")
        if exists(cache_fn):
            with open(cache_fn, 'r') as fp:
                data = json.load(fp)
                self.data_file_paths = data['file_paths']
                self.targets = data['targets']
        else:
            
            # Grab all relevant file paths
            self.data_file_paths = get_paths(data_folder, extensions=('.npy'))
            
            # Open up labels path
            labels = pd.read_csv(labels_path, dtype={'id': str, 'target':int})
            labels = dict(labels.values)
            
            # Iterate through each label file, accumulate the label for it
            self.targets = []
            for file in self.data_file_paths:
                
                # Grab just the basename, no extension
                file_key = basename(file).split('.')[0]
                self.targets.extend(labels[file_key])
                
            
            # Cache for later use
            with open(cache_fn, 'w') as fp:
                json.dump({'file_paths':self.data_file_paths, 'targets':self.targets}, fp)
        
        
        def __len__(self):
            return len(self.targets)
        
        def __getitem__(self, idx):
            
            # Read file at given index
            data = np.load(self.data_file_paths[idx])
            data = data.astype(np.float32)
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
        