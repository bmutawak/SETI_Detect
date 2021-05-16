import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import plot_one_cadence


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
    

def main():
    
    # Load a sample image
    fp = '/home/bassam/Desktop/dev/kaggle/SETI_Detect/data/train/0/0030dc7cf6e0.npy'
    data = np.load(fp)
    
    # Display
    plot_one_cadence(data)
    
    return





if __name__=="__main__":
    main()
    