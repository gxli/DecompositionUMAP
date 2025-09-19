import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# --- Add parent directory to path to find the 'src' package ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Import your package ---
import src as decomposition_umap

# --- Import required library for 3D plotting ---
from mpl_toolkits.mplot3d import Axes3D

# --- Data Generation and Plotting Functions ---

def generate_pink_noise(shape):
    """Generates a 2D array of pink noise."""
    rows, cols = shape
    u, v = np.fft.fftfreq(rows), np.fft.fftfreq(cols)
    frequency_radius = np.sqrt(u[:, np.newaxis]**2 + v**2)
    frequency_radius[0, 0] = 1.0
    fft_white_noise = np.fft.fft2(np.random.randn(rows, cols))
    fft_pink_noise = fft_white_noise / frequency_radius
    pink_noise = np.real(np.fft.ifft2(fft_pink_noise))
    return (pink_noise - pink_noise.mean()) / pink_noise.std()

def add_gaussian_blobs(data, centers, sigmas, amplitudes):
    """Adds one or more 2D Gaussian blobs to an existing data array."""
    rows, cols = data.shape
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    signal = np.zeros_like(data, dtype=float)
    for center, sigma, amp in zip(centers, sigmas, amplitudes):
        cx, cy, sx, sy = *center, *sigma
        signal += amp * np.exp(-(((x - cx)**2 / (2 * sx**2)) + ((y - cy)**2 / (2 * sy**2))))
    return data + signal, signal

def plot_original_data(title, data, noise, signal):
    """Generates a figure showing the original input data components."""
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

def plot_multivariate_data(title, data):
    """Generates a figure showing each channel of a multivariate dataset."""
    num_channels = data.shape[0]
    fig, axes = plt.subplots(1, num_channels, figsize=(num_channels * 6, 5))
    if num_channels == 1: axes = [axes]
    for i in range(num_channels):
        im = axes[i].imshow(data[i], cmap='viridis', origin='lower')
        axes[i].set_title(f'Input Channel {i+1}')
        fig.colorbar(im, ax=axes[i])
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

def plot_umap_embedding_scatter_3d(title, embed_map, signal_blobs):
    """Generates a 3D scatter plot of the UMAP embedding."""
    umap_x, umap_y, umap_z = [em.flatten() for em in embed_map]
    if signal_blobs.ndim > 2:
        is_signal = np.sum(signal_blobs, axis=0).flatten() > 0.1
    else:
        is_signal = signal_blobs.flatten() > 0.1
    noise_x, noise_y, noise_z = umap_x[~is_signal], umap_y[~is_signal], umap_z[~is_signal]
    signal_x, signal_y, signal_z = umap_x[is_signal], umap_y[is_signal], umap_z[is_signal]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(noise_x, noise_y, noise_z, label='Noise', alpha=0.05, s=10, color='gray')
    ax.scatter(signal_x, signal_y, signal_z, label='Signal (Blobs)', alpha=0.8, s=15, color='red')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('UMAP Dim 1'); ax.set_ylabel('UMAP Dim 2'); ax.set_zlabel('UMAP Dim 3')
    ax.legend(); ax.grid(True)

# --- Main execution block ---
if __name__ == '__main__':
    shape = (256, 256)

    # --- Example 1: Single Dataset Mode (`data=...`) ---
    print("--- Running Example 1: Single Dataset Mode ---")
    data_1, signal_1 = add_gaussian_blobs(generate_pink_noise(shape),
                                          centers=[(60, 80)], sigmas=[(15, 15)], amplitudes=[5.0])
    embed_map, _, _ = decomposition_umap.decompose_and_embed(
        data=data_1, # Use the 'data' keyword
        decomposition_max_n=6, n_component=3, verbose=True
    )
    plot_original_data('Single Dataset Input', data_1, data_1 - signal_1, signal_1)
    plot_umap_embedding_scatter_3d('Single Dataset Embedding', embed_map, signal_1)

    # --- Example 2: Multi-Dataset Mode (`datasets=...`) ---
    print("\n--- Running Example 2: Multi-Dataset (Batch) Mode ---")
    data_2, signal_2 = add_gaussian_blobs(generate_pink_noise(shape),
                                          centers=[(160, 180)], sigmas=[(10, 10)], amplitudes=[4.0])
    list_of_embeds, _, _ = decomposition_umap.decompose_and_embed(
        datasets=[data_1, data_2], # Use the 'datasets' keyword
        decomposition_max_n=6, n_component=3, verbose=True
    )
    print(f"Multi-Dataset mode returned {len(list_of_embeds)} embedding maps.")
    # Plot the result for the second dataset from the batch
    plot_umap_embedding_scatter_3d('Multi-Dataset Embedding (Result for Data 2)', list_of_embeds[1], signal_2)

    # --- Example 3: Multivariate Mode (`data_multivariate=...`) ---
    print("\n--- Running Example 3: Multivariate Mode ---")
    channel_1, signal_c1 = data_1, signal_1
    channel_2, signal_c2 = data_2, signal_2
    channel_3 = generate_pink_noise(shape)
    
    multivariate_data = np.array([channel_1, channel_2, channel_3])
    combined_signal = np.array([signal_c1, signal_c2, np.zeros_like(signal_c1)])
    
    embed_map_multi, decomp_multi, _ = decomposition_umap.decompose_and_embed(
        data_multivariate=multivariate_data, # Use the 'data_multivariate' keyword
        decomposition_max_n=4, n_component=3, verbose=True
    )
    print(f"Multivariate mode produced a merged decomposition of shape: {decomp_multi.shape}")
    plot_multivariate_data('Multivariate Input Channels', multivariate_data)
    plot_umap_embedding_scatter_3d('Multivariate Embedding', embed_map_multi, combined_signal)
    
    print("\n--- All examples finished. Displaying plots. ---")
    plt.show()