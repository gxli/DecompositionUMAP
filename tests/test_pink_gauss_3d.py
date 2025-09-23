import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# --- Add parent directory to path to find the 'src' package ---
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Import your package ---
import src as decomposition_umap

# --- Import required library for 3D plotting ---
from mpl_toolkits.mplot3d import Axes3D

# --- 1. Data Generation ---

def generate_pink_noise(shape):
    """Generates pink noise with a 1/f power spectrum."""
    rows, cols = shape
    u, v = np.fft.fftfreq(rows), np.fft.fftfreq(cols)
    frequency_radius = np.sqrt(u[:, np.newaxis]**2 + v**2)
    frequency_radius[0, 0] = 1.0
    fft_white_noise = np.fft.fft2(np.random.randn(rows, cols))
    fft_pink_noise = fft_white_noise / frequency_radius
    pink_noise = np.real(np.fft.ifft2(fft_pink_noise))
    return (pink_noise - pink_noise.mean()) / pink_noise.std()

def add_gaussian_blobs(data, centers, sigmas, amplitudes):
    """Adds Gaussian blobs to an existing data array."""
    rows, cols = data.shape
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    signal = np.zeros_like(data, dtype=float)
    for center, sigma, amp in zip(centers, sigmas, amplitudes):
        cx, cy = center
        sx, sy = sigma
        signal += amp * np.exp(-(((x - cx)**2 / (2 * sx**2)) + ((y - cy)**2 / (2 * sy**2))))
    return data + signal, signal

# --- 2. Plotting Functions ---

def plot_original_data(title, data, noise, signal):
    """Generates a figure showing the noise, the signal, and the combined data."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    im1 = axes[0].imshow(noise, cmap='gray', origin='lower')
    axes[0].set_title('Pink Noise Background')
    fig.colorbar(im1, ax=axes[0], label='Amplitude')
    im2 = axes[1].imshow(signal, cmap='gray', origin='lower')
    axes[1].set_title('Gaussian Blobs (Signal)')
    fig.colorbar(im2, ax=axes[1], label='Amplitude')
    im3 = axes[2].imshow(data, cmap='viridis', origin='lower')
    axes[2].set_title('Final Data (Noise + Signal)')
    fig.colorbar(im3, ax=axes[2], label='Amplitude')
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

def plot_decomposition_components(title, decomposition):
    """Generates a figure showing all decomposition components in a grid."""
    num_components = decomposition.shape[0]
    ncols = int(np.ceil(np.sqrt(num_components)))
    nrows = int(np.ceil(num_components / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = axes.flatten()
    for i in range(num_components):
        im = axes[i].imshow(decomposition[i], cmap='viridis', origin='lower')
        axes[i].set_title(f'Component {i+1}')
        fig.colorbar(im, ax=axes[i])
    for j in range(num_components, len(axes)):
        axes[j].axis('off')
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

def plot_umap_embedding_scatter_2d(title, embed_map, signal_blobs):
    """Generates a 2D scatter plot of the UMAP embedding."""
    umap_x = embed_map[0].flatten()
    umap_y = embed_map[1].flatten()
    is_signal = signal_blobs.flatten() > 2

    noise_x, noise_y = umap_x[~is_signal], umap_y[~is_signal]
    signal_x, signal_y = umap_x[is_signal], umap_y[is_signal]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(noise_x, noise_y, label='Noise', alpha=0.05, s=10, color='gray')
    ax.scatter(signal_x, signal_y, label='Signal (Blobs)', alpha=0.8, s=15, color='red')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    ax.legend()
    ax.grid(True)

def plot_umap_embedding_scatter_3d(title, embed_map, signal_blobs):
    """Generates a 3D scatter plot of the UMAP embedding."""
    umap_x = embed_map[0].flatten()
    umap_y = embed_map[1].flatten()
    umap_z = embed_map[2].flatten()
    is_signal = signal_blobs.flatten() > 0.2

    noise_x, noise_y, noise_z = umap_x[~is_signal], umap_y[~is_signal], umap_z[~is_signal]
    signal_x, signal_y, signal_z = umap_x[is_signal], umap_y[is_signal], umap_z[is_signal]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(noise_x, noise_y, noise_z, label='Noise', alpha=0.05, s=10, color='gray')
    ax.scatter(signal_x, signal_y, signal_z, label='Signal (Blobs)', alpha=0.8, s=15, color='red')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    ax.set_zlabel('UMAP Dimension 3')
    ax.legend()
    ax.grid(True)

# --- 3. Main execution block ---

if __name__ == '__main__':
    # --- Generate the common dataset for all examples ---
    print("--- Generating synthetic data: Gaussian blobs in pink noise ---")
    shape = (256, 256)
    pink_noise = generate_pink_noise(shape)
    data, signal_blobs = add_gaussian_blobs(
        pink_noise,
        centers=[(60, 80), (160, 180), (100, 200)],
        sigmas=[(10, 10), (16, 8), (12, 12)],
        amplitudes=[5.0, 3.0, 3.0]
    )

    # --- Output directory ---
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # --- 2D Example ---
    print("\n--- Running decompose_and_embed for 2D embedding ---")
    embed_map_2d, decomposition_2d, umap_model_2d = decomposition_umap.decompose_and_embed(
        data,
        decomposition_method='cdd',
        decomposition_max_n=6,
        n_component=2,  # 2D embedding
        verbose=True,
    )

    # Save 2D results
    print("\n--- Saving 2D results to .npy files ---")
    np.save(os.path.join(output_dir, "original_data_2d.npy"), data)
    np.save(os.path.join(output_dir, "decomposition_2d.npy"), decomposition_2d)
    np.save(os.path.join(output_dir, "embed_map_x_2d.npy"), embed_map_2d[0])
    np.save(os.path.join(output_dir, "embed_map_y_2d.npy"), embed_map_2d[1])

    # --- 3D Example ---
    print("\n--- Running decompose_and_embed for 3D embedding ---")
    embed_map_3d, decomposition_3d, umap_model_3d = decomposition_umap.decompose_and_embed(
        data,
        decomposition_method='cdd',
        decomposition_max_n=6,
        n_component=3,  # 3D embedding
        verbose=True,
    )

    # Save 3D results
    print("\n--- Saving 3D results to .npy files ---")
    np.save(os.path.join(output_dir, "original_data_3d.npy"), data)
    np.save(os.path.join(output_dir, "decomposition_3d.npy"), decomposition_3d)
    np.save(os.path.join(output_dir, "embed_map_x_3d.npy"), embed_map_3d[0])
    np.save(os.path.join(output_dir, "embed_map_y_3d.npy"), embed_map_3d[1])
    np.save(os.path.join(output_dir, "embed_map_z_3d.npy"), embed_map_3d[2])

    print(f"Results saved in '{output_dir}/' directory.")

    # --- Generate plots ---
    print("\n--- Generating plots ---")
    # Common data plots
    plot_original_data('Input Data Components', data, pink_noise, signal_blobs)
    # 2D example plots
    plot_decomposition_components('Decomposition Components via CDD (2D)', decomposition_2d)
    plot_umap_embedding_scatter_2d('2D UMAP Embedding Scatter Plot', embed_map_2d, signal_blobs)
    # 3D example plots
    plot_decomposition_components('Decomposition Components via CDD (3D)', decomposition_3d)
    plot_umap_embedding_scatter_3d('3D UMAP Embedding Scatter Plot', embed_map_3d, signal_blobs)

    # Show all generated figures
    plt.show()