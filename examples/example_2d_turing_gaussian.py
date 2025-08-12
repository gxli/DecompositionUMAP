import matplotlib.pyplot as plt
import decomposition_umap
import numpy as np
import math

def plot_images(p, figsize_per_subplot=3):
    """
    Plot all images in array p using subplots, arranged in a grid.
    
    Parameters:
    p (list or array): List of images (e.g., NumPy arrays compatible with imshow).
    figsize_per_subplot (float): Size per subplot (default: 3 inches).
    
    Returns:
    None
    """
    n = len(p)  # Number of images
    if n == 0:
        print("No images to plot.")
        return

    # Calculate grid dimensions (try to make it as square as possible)
    cols = math.ceil(math.sqrt(n))  # Number of columns
    rows = math.ceil(n / cols)      # Number of rows

    # Create a figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_per_subplot, rows * figsize_per_subplot))

    # Flatten axes array for easy iteration (handles both 1D and 2D cases)
    axes = np.array(axes).flatten()

    # Plot each image
    for i in range(n):
        axes[i].imshow(p[i])  # Plot the i-th image
        axes[i].axis('off')   # Hide axes for cleaner display
        axes[i].set_title(f'Image {i+1}')  # Optional: add a title

    # Turn off unused subplots
    for i in range(n, len(axes)):
        axes[i].axis('off')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()



data, signal, anomaly =decomposition_umap.example.generate_turing_with_gaussian()

plt.figure()
plt.imshow(data, cmap='gray')
plt.show()
# embedding, decomposition, umap_obj = decomposition_umap.decompose_and_embed(data)

embedding, decomposition, umap_obj = decomposition_umap.decompose_and_embed(data,decomposition_method='amd')

datax = embedding[0]

datay = embedding[1]


plt.figure()
plot_images(decomposition)

plt.figure()
plt.scatter(datax.flatten(), datay.flatten(), s=1, c='blue', alpha=0.5)

plt.show()

# Save datax and datay as .npy files with 'turing_' prefix
np.save('turing_data.npy', data)
np.save('turing_signal.npy', signal)
np.save('turing_anomaly.npy', anomaly)
np.save('turing_umap_x.npy', datax)
np.save('turing_umap_y.npy', datay)
