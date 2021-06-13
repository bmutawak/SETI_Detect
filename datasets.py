import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
import torchvision
import cv2
import numpy as np
from os.path import join, basename, dirname, exists
import json
from utils import get_paths, get_files_paths_and_labels
from utils import get_validation_augmentations, get_training_augmentations
import pandas as pd
from sklearn.model_selection import train_test_split

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
        return
    
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        # Read file at given index
        data = np.load(self.data_file_paths[idx])
        data = data.astype(np.float32)
        data = torch.from_numpy(data)
        
        # Perform augmentations if desired
        if not self.transform is None:

            data = self.transform(data)

        else:
            data = data[np.newaxis, :, :]
            data = torch.from_numpy(data).float()
        
        # Grab label, return
        label = torch.tensor(self.targets[idx]).float()
        return data, label


def get_dataloaders(data_dir, hyp):
    """
    Ingests the data folder and returns training and validation data loaders.

    Parameters
    ----------
    data_dir : path
        Path to parent data directory.
    hyp : TYPE
        hyperparameters desired.

    Returns
    -------
    Train, validation dataloaders.
    
    """
    
    # Grab data, targets
    data_file_paths, targets = get_files_paths_and_labels(data_dir)
    
    # Split into train/validation
    train_data, val_data, train_labels, val_labels = train_test_split(data_file_paths,
                                                                      targets,
                                                                      train_size=hyp['perc_train'],
                                                                      shuffle=hyp['shuffle'],
                                                                      stratify=targets)

    # Create train/validation augmentation handler
    train_aug = get_training_augmentations(hyp)
    val_aug = get_validation_augmentations(hyp)
    
    # Create datasets
    train_dset = SETIDataset(train_data, train_labels, transform=train_aug)
    val_dset = SETIDataset(val_data, val_labels, transform=val_aug)
    
    # Create dataloaders
    train_loader = DataLoader(train_dset, shuffle=True, batch_size=hyp['batch_size'], 
                              pin_memory=True, num_workers=8)
    
    val_loader = DataLoader(val_dset, batch_size=hyp['batch_size'], 
                              pin_memory=True, num_workers=8)
    
    return train_loader, val_loader
    
    
def get_dataset_parameters(dataloader):
    """
    Returns mean, std of data.

    Parameters
    ----------
    dataloader : torch.utils.DataLoader
        dataset loader.

    Returns
    -------
    None.

    """
    mean = 0.0
    meansq = 0.0
    count = 0
    
    for index, (data, targets) in enumerate(dataloader):
        mean = data.sum()
        meansq = meansq + (data**2).sum()
        count += np.prod(data.shape)
    
    total_mean = mean/count
    total_var = (meansq/count) - (total_mean**2)
    total_std = torch.sqrt(total_var)
    print("mean: " + str(total_mean))
    print("std: " + str(total_std))
    
    
    
    
    
    