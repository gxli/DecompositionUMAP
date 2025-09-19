import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
import os
import sys
import time

# Add parent directory to path (handle interactive environments)
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
except NameError:
    sys.path.append(os.path.abspath('..'))

# Import custom package
try:
    import src as decomposition_umap
except ImportError as e:
    print(f"Error importing decomposition_umap: {e}")
    sys.exit(1)

def plot_images(p, figsize_per_subplot=3, title="Decomposition Components"):
    """
    Plot all images in array p using subplots, arranged in a grid.
    
    Parameters:
    p (list or array): List of images (e.g., NumPy arrays compatible with imshow).
    figsize_per_subplot (float): Size per subplot (default: 3 inches).
    title (str): Super-title for the figure.
    
    Returns:
    fig, axes: Figure and axes objects for deferred plotting
    """
    if p is None or len(p) == 0:
        print("No images to plot.")
        return None, None

    n = len(p)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_per_subplot, rows * figsize_per_subplot))
    axes = np.array(axes).flatten()

    for i in range(n):
        axes[i].imshow(p[i])
        axes[i].axis('off')
        axes[i].set_title(f'Component {i+1}')
    for i in range(n, len(axes)):
        axes[i].axis('off')

    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, axes

# Generate example data
try:
    data, signal, anomaly = decomposition_umap.example.generate_turing_with_gaussian()
except Exception as e:
    print(f"Error generating data: {e}")
    sys.exit(1)

# Debug shapes
print(f"Data shape: {data.shape}")
print(f"Signal shape: {signal.shape}")
print(f"Anomaly shape: {anomaly.shape}")

# Create figure for original data
fig1 = plt.figure(figsize=(6, 6))
plt.imshow(data, cmap='gray')
plt.title('Original Data')
plt.axis('off')

# Perform 2D UMAP embedding
try:
    embedding_2d, decomposition_2d, umap_obj_2d = decomposition_umap.decompose_and_embed(
        data, decomposition_method='amd', n_component=2
    )
    # Print shapes of embedding_2d elements
    print(f"2D embedding[0] shape: {embedding_2d[0].shape}")
    print(f"2D embedding[1] shape: {embedding_2d[1].shape}")
    # Print shapes of decomposition_2d
    if isinstance(decomposition_2d, list):
        for i, dec in enumerate(decomposition_2d):
            print(f"2D decomposition[{i}] shape: {dec.shape}")
    else:
        print(f"2D decomposition shape: {decomposition_2d.shape}")
except Exception as e:
    print(f"Error in 2D UMAP: {e}")
    sys.exit(1)

# Extract 2D UMAP embedding coordinates
datax_2d = embedding_2d[0]
datay_2d = embedding_2d[1]

# Perform 3D UMAP embedding
try:
    embedding_3d, decomposition_3d, umap_obj_3d = decomposition_umap.decompose_and_embed(
        data, decomposition_method='amd', n_component=3
    )
    # Print shapes of embedding_3d elements
    print(f"3D embedding[0] shape: {embedding_3d[0].shape}")
    print(f"3D embedding[1] shape: {embedding_3d[1].shape}")
    print(f"3D embedding[2] shape: {embedding_3d[2].shape}")
    # Print shapes of decomposition_3d
    if isinstance(decomposition_3d, list):
        for i, dec in enumerate(decomposition_3d):
            print(f"3D decomposition[{i}] shape: {dec.shape}")
    else:
        print(f"3D decomposition shape: {decomposition_3d.shape}")
except Exception as e:
    print(f"Error in 3D UMAP: {e}")
    sys.exit(1)

# Extract 3D UMAP embedding coordinates
datax_3d = embedding_3d[0]
datay_3d = embedding_3d[1]
dataz_3d = embedding_3d[2]

# Flatten anomaly and signal, and ensure alignment
anomaly_flat = anomaly.flatten() if anomaly.ndim > 1 else anomaly
signal_flat = signal.flatten() if signal.ndim > 1 else signal
min_len_2d = min(len(datax_2d.flatten()), len(datay_2d.flatten()), len(anomaly_flat), len(signal_flat))
datax_2d_flat = datax_2d.flatten()[:min_len_2d]
datay_2d_flat = datay_2d.flatten()[:min_len_2d]
anomaly_flat_2d = anomaly_flat[:min_len_2d]
signal_flat_2d = signal_flat[:min_len_2d]

min_len_3d = min(len(datax_3d.flatten()), len(datay_3d.flatten()), len(dataz_3d.flatten()), len(anomaly_flat), len(signal_flat))
datax_3d_flat = datax_3d.flatten()[:min_len_3d]
datay_3d_flat = datay_3d.flatten()[:min_len_3d]
dataz_3d_flat = dataz_3d.flatten()[:min_len_3d]
anomaly_flat_3d = anomaly_flat[:min_len_3d]
signal_flat_3d = signal_flat[:min_len_3d]

# Create figure for 2D scatter plot
fig2 = plt.figure(figsize=(8, 6))
# Compute anomaly/signal ratio, avoiding division by zero
ratio_2d = anomaly_flat_2d / (signal_flat_2d + 1e-10)
anomaly_mask_2d = ratio_2d > 0.5
normal_mask_2d = ~anomaly_mask_2d

plt.scatter(
    datax_2d_flat[normal_mask_2d],
    datay_2d_flat[normal_mask_2d],
    c='blue',
    s=10,
    alpha=0.5,
    label='Normal Points'
)
plt.scatter(
    datax_2d_flat[anomaly_mask_2d],
    datay_2d_flat[anomaly_mask_2d],
    c='red',
    s=20,
    alpha=0.8,
    label='Anomalies (anomaly/signal > 0.5)'
)
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.title('2D UMAP Embedding with Gaussian Anomalies')
plt.legend()

# Create figure for 3D scatter plot
fig3 = plt.figure(figsize=(10, 8))
ax = fig3.add_subplot(111, projection='3d')
# Compute anomaly/signal ratio for 3D
ratio_3d = anomaly_flat_3d / (signal_flat_3d + 1e-10)
anomaly_mask_3d = ratio_3d > 0.5
normal_mask_3d = ~anomaly_mask_3d

ax.scatter(
    datax_3d_flat[normal_mask_3d],
    datay_3d_flat[normal_mask_3d],
    dataz_3d_flat[normal_mask_3d],
    c='blue',
    s=10,
    alpha=0.5,
    label='Normal Points'
)
ax.scatter(
    datax_3d_flat[anomaly_mask_3d],
    datay_3d_flat[anomaly_mask_3d],
    dataz_3d_flat[anomaly_mask_3d],
    c='red',
    s=20,
    alpha=0.8,
    label='Anomalies (anomaly/signal > 0.5)'
)
ax.set_xlabel('UMAP Dimension 1')
ax.set_ylabel('UMAP Dimension 2')
ax.set_zlabel('UMAP Dimension 3')
ax.set_title('3D UMAP Embedding with Gaussian Anomalies')
ax.legend()

# Plot decomposition images
fig4, _ = plot_images(decomposition_2d, title='2D Decomposition Components')
fig5, _ = plot_images(decomposition_3d, title='3D Decomposition Components')

# Save data and embeddings
prefix = f'results/turing_{int(time.time())}_'  # Unique prefix to avoid overwriting
os.makedirs('results', exist_ok=True)
try:
    np.save(f'{prefix}data.npy', data)
    np.save(f'{prefix}signal.npy', signal)
    np.save(f'{prefix}anomaly.npy', anomaly)
    np.save(f'{prefix}umap_x_2d.npy', datax_2d)
    np.save(f'{prefix}umap_y_2d.npy', datay_2d)
    np.save(f'{prefix}umap_x_3d.npy', datax_3d)
    np.save(f'{prefix}umap_y_3d.npy', datay_3d)
    np.save(f'{prefix}umap_z_3d.npy', dataz_3d)
except Exception as e:
    print(f"Error saving data: {e}")

# Save figures
try:
    fig1.savefig(f'{prefix}original_data.png', bbox_inches='tight')
    fig2.savefig(f'{prefix}umap_2d.png', bbox_inches='tight')
    fig3.savefig(f'{prefix}umap_3d.png', bbox_inches='tight')
    if fig4 is not None:
        fig4.savefig(f'{prefix}decomposition_2d.png', bbox_inches='tight')
    if fig5 is not None:
        fig5.savefig(f'{prefix}decomposition_3d.png', bbox_inches='tight')
except Exception as e:
    print(f"Error saving figures: {e}")

# Display all plots
plt.show()