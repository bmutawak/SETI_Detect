import matplotlib.pyplot as plt
import cv2


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

    
    
    
    