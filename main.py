import numpy as np
import cv2






def main():
    
    # Load a sample image
    fp = '/home/bassam/Desktop/dev/kaggle/SETI_Detect/data/train/1/1a0fc0743024.npy'
    data = np.load(fp)
    
    # Create window
    cv2.namedWindow('image')
    
    # Display
    cv2.imshow('image', data[0, :, :].astype(np.single))
    cv2.waitKey(0)
    
    return





if __name__=="__main__":
    main()
    