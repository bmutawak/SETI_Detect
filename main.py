import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import plot_one_cadence




def main():
    
    # Load a sample image
    fp = '/home/bassam/Desktop/dev/kaggle/SETI_Detect/data/train/1/1a0fc0743024.npy'
    data = np.load(fp)
    
    # Display
    plot_one_cadence(data)
    
    return





if __name__=="__main__":
    main()
    