import matplotlib.pyplot as plt
import cv2
import os
from os.path import join, basename, dirname, exists
import torch
from torchvision import transforms
import json
import pandas as pd


def get_paths(folder_path, recurse=True, extensions=''):
    """
    Grabs all relevant file paths from folder_path with extension.

    Parameters
    ----------
    folder_path : PATH-STR
        Path to parent folder.
    recurse : BOOL
        T/F if we want to recurse all subdirectories
    extension : TUPLE, optional
        Exclusive tuple of extensions. The default is None.

    Returns
    -------
    List of absolute paths.
    
    """
    
    # Make sure directory exists
    if not  os.path.exists(folder_path):
        print('ERROR: FOLDER_PATH NOT FOUND')
        return []
    
    file_paths = []
    if recurse:
        for folder, subs, files in os.walk(folder_path):
            file_paths.extend([os.path.join(folder, file) for file in files if file.endswith(extensions)])
    else:
        file_paths = [os.path.join(folder_path, path) for path in os.listdir(folder_path) if path.endswith(extensions)]
    
    
    return file_paths

def plot_one_cadence(cadence, cmap = 'plasma'):
    """
    Plots one "cadence" sample from the SETI dataset.
    
    Parameters
    ----------
    cadence : NUMPY ARRAY
        Cadence array.
    cmap : STRING
        colormap string if desired.

    Returns
    -------
    None.

    """
    
    # Grab number of snippets, create subplot
    num_snippets = cadence.shape[0]
    plt.figure()
    plt.suptitle('Cadence')
    plt.xlabel('Frequency')
    
    # Loop through each snippet, plotting
    for snippet in range(num_snippets):
        plt.subplot(num_snippets, 1, snippet + 1)
        plt.imshow(cadence[snippet, :, :].astype(float), cmap=cmap, aspect='auto')
        plt.text(5, 100, ['ON', 'OFF'][snippet % 2], bbox={'facecolor': 'white'})
        plt.xticks([])
        
        
    plt.show()
    return


def get_training_augmentations(image_size, rotation_degrees, horizontal_flip_prob, vertical_flip_prob):
    """
    Creates a function to perform all training image augmentations. 

    Returns
    -------
    Function.

    """

    # Create sequential transforms
    augmentations = torch.nn.Sequential(
        transforms.Resize(image_size),
        transforms.RandomRotation(rotation_degrees),
        transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
        transforms.RandomVerticalFlip(p=vertical_flip_prob),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    )   

    return augmentations

def get_validation_augmentations(image_size):
    """
    Creates a function to perform all validation augmentations

    Parameters
    ----------
    image_size : INT
        Size of image.

    Returns
    -------
    Function.

    """
    
    # Create sequential transforms
    augmentations = torch.nn.Sequential(
        transforms.Resize(image_size),
        transforms.ToTensor())
    
    return augmentations

def get_files_paths_and_labels(data_folder):
    """
    Gets all file paths for data and labels.
    
    Parameters
    ----------
    data_folder : Path, STR
        Path to training folder.

    Returns
    -------
    data_file_paths : LIST
        List of paths to data.
    targets : LIST
        List of target values.

    """
    # Check for cached file names
    cache_fn = join(data_folder, "file_paths.json")
    if exists(cache_fn):
        with open(cache_fn, 'r') as fp:
            data = json.load(fp)
            data_file_paths = data['file_paths']
            targets = data['targets']
    else:
        
        # Grab all relevant file paths
        data_file_paths = get_paths(data_folder, extensions=('.npy'))
        
        # Open up labels path
        labels_path = join(dirname(data_folder), "train_labels.csv")
        labels = pd.read_csv(labels_path, dtype={'id': str, 'target':int})
        labels = dict(labels.values)
        
        # Iterate through each label file, accumulate the label for it
        targets = []
        for file in data_file_paths:
            
            # Grab just the basename, no extension
            file_key = basename(file).split('.')[0]
            targets.extend(labels[file_key])
            
        
        # Cache for later use
        with open(cache_fn, 'w') as fp:
            json.dump({'file_paths':data_file_paths, 'targets':targets}, fp)
    
    return data_file_paths, targets