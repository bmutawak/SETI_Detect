import matplotlib.pyplot as plt
import cv2
import os




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

    
    
    
    