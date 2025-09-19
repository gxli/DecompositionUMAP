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

def plot_umap_embedding_scatter_3d(title, embed_map, signal_blobs):
    """Generates a 3D scatter plot of the UMAP embedding."""
    umap_x, umap_y, umap_z = [em.flatten() for em in embed_map]
    is_signal = signal_blobs.flatten() > 0.1

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
    # --- Generate two different datasets to demonstrate the multi-dataset feature ---
    print("--- Generating two synthetic datasets ---")
    shape = (256, 256)
    
    # Create Dataset 1
    pink_noise_1 = generate_pink_noise(shape)
    data_1, signal_blobs_1 = add_gaussian_blobs(
        pink_noise_1,
        centers=[(60, 80), (160, 180)],
        sigmas=[(10, 10), (16, 8)],
        amplitudes=[4.0, 3.5]
    )

    # Create Dataset 2 with a different signal
    pink_noise_2 = generate_pink_noise(shape)
    data_2, signal_blobs_2 = add_gaussian_blobs(
        pink_noise_2,
        centers=[(100, 200), (200, 70)],
        sigmas=[(15, 15), (10, 20)],
        amplitudes=[5.0, 4.0]
    )

    # --- Run the Multi-Dataset UMAP Pipeline ---
    print("\n--- Running decompose_and_embed in multi-dataset mode ---")
    # A single UMAP model is trained on the combined features from both datasets.
    list_of_embed_maps, list_of_decompositions, umap_model = decomposition_umap.decompose_and_embed(
        [data_1, data_2],  # The input is a list of arrays
        decomposition_method='cdd',
        decomposition_max_n=6,
        n_component=3,
        verbose=True,
    )
    print(f"\nProcessing complete. Received {len(list_of_embed_maps)} embedding maps.")

    # --- Process and save results for Dataset 1 ---
    print("\n--- Processing and saving results for Dataset 1 ---")
    output_dir_1 = "results_dataset_1"
    os.makedirs(output_dir_1, exist_ok=True)
    
    embed_map_1 = list_of_embed_maps[0]
    decomposition_1 = list_of_decompositions[0]
    
    np.save(os.path.join(output_dir_1, "original_data.npy"), data_1)
    np.save(os.path.join(output_dir_1, "decomposition.npy"), decomposition_1)
    np.save(os.path.join(output_dir_1, "embed_map_x.npy"), embed_map_1[0])
    np.save(os.path.join(output_dir_1, "embed_map_y.npy"), embed_map_1[1])
    np.save(os.path.join(output_dir_1, "embed_map_z.npy"), embed_map_1[2])
    print(f"Results for Dataset 1 saved in '{output_dir_1}/'")

    # --- Process and save results for Dataset 2 ---
    print("\n--- Processing and saving results for Dataset 2 ---")
    output_dir_2 = "results_dataset_2"
    os.makedirs(output_dir_2, exist_ok=True)

    embed_map_2 = list_of_embed_maps[1]
    decomposition_2 = list_of_decompositions[1]

    np.save(os.path.join(output_dir_2, "original_data.npy"), data_2)
    np.save(os.path.join(output_dir_2, "decomposition.npy"), decomposition_2)
    np.save(os.path.join(output_dir_2, "embed_map_x.npy"), embed_map_2[0])
    np.save(os.path.join(output_dir_2, "embed_map_y.npy"), embed_map_2[1])
    np.save(os.path.join(output_dir_2, "embed_map_z.npy"), embed_map_2[2])
    print(f"Results for Dataset 2 saved in '{output_dir_2}/'")

    # --- Generate plots for both datasets ---
    print("\n--- Generating plots for Dataset 1 ---")
    plot_original_data('Input Data (Dataset 1)', data_1, pink_noise_1, signal_blobs_1)
    plot_decomposition_components('Decomposition (Dataset 1)', decomposition_1)
    plot_umap_embedding_scatter_3d('3D UMAP Embedding (Dataset 1)', embed_map_1, signal_blobs_1)
    
    print("\n--- Generating plots for Dataset 2 ---")
    plot_original_data('Input Data (Dataset 2)', data_2, pink_noise_2, signal_blobs_2)
    plot_decomposition_components('Decomposition (Dataset 2)', decomposition_2)
    plot_umap_embedding_scatter_3d('3D UMAP Embedding (Dataset 2)', embed_map_2, signal_blobs_2)

    # Show all generated figures at the end
    print("\n--- Displaying all plots ---")
    plt.show()