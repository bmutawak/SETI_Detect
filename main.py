import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import plot_one_cadence
from datasets import SETIDataset



# Relevant hyperparamters
hyp = {
       'image_size':256,
       'batch_size':16,
       'vertical_flip_prob':0.5,
       'horizontal_flip_prob':0.5,
       'rotation_degrees':25,
       'lr':0.001,
       'adam':False,
       
       
       
       }
def process_cadence(cadence):
    """
    Preprocesses a cadence to reduce backgrond noise 

    Parameters
    ----------
    cadence : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    return


def main():
    
    # Paths to relevant files
    train_dir = './data/train'
    labels = './data/train_labels.csv'
    
    
    dataset = SETIDataset(train_dir, labels)
    
    dataset[1]
    
    
    return





if __name__=="__main__":
    main()
    