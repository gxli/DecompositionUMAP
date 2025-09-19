import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# --- Add parent directory to path to find the 'src' package ---
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Import your package ---
import src as decomposition_umap

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
    """
    Generates a figure showing the noise, the signal, and the combined data.
    """
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # Plot 1: Pink Noise Background
    im1 = axes[0].imshow(noise, cmap='gray', origin='lower')
    axes[0].set_title('Pink Noise Background')
    fig.colorbar(im1, ax=axes[0], label='Amplitude')

    # Plot 2: Gaussian Blobs Signal
    im2 = axes[1].imshow(signal, cmap='gray', origin='lower')
    axes[1].set_title('Gaussian Blobs (Signal)')
    fig.colorbar(im2, ax=axes[1], label='Amplitude')

    # Plot 3: Combined Data
    im3 = axes[2].imshow(data, cmap='viridis', origin='lower')
    axes[2].set_title('Final Data (Noise + Signal)')
    fig.colorbar(im3, ax=axes[2], label='Amplitude')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()

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
    # plt.show()

def plot_umap_embedding_scatter(title, embed_map, signal_blobs):
    """
    Generates a scatter plot of the UMAP embedding, highlighting signal points.
    """
    umap_x = embed_map[0].flatten()
    umap_y = embed_map[1].flatten()
    is_signal = signal_blobs.flatten() > 0.1

    noise_x, noise_y = umap_x[~is_signal], umap_y[~is_signal]
    signal_x, signal_y = umap_x[is_signal], umap_y[is_signal]

    plt.figure(figsize=(8, 8))
    plt.scatter(noise_x, noise_y, label='Noise', alpha=0.1, s=10, color='gray')
    plt.scatter(signal_x, signal_y, label='Signal (Blobs)', alpha=0.8, s=15, color='red')
    plt.title(title, fontsize=16)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')
    plt.tight_layout()
    # plt.show()

# --- 3. Main execution block ---

if __name__ == '__main__':
    # --- Generate the common dataset for all examples ---
    print("--- Generating synthetic data: Gaussian blobs in pink noise ---")
    shape = (256, 256)
    pink_noise = generate_pink_noise(shape)
    data, signal_blobs = add_gaussian_blobs(
        pink_noise,
        centers=[(30, 40), (80, 90), (50, 100)],
        sigmas=[(5, 5), (8, 4), (6, 6)],
        amplitudes=[5.0, 3, 3]
    )

    # --- Example 1: High-Level API `decompose_and_embed` ---
    print("\n--- Running Example 1: High-level `decompose_and_embed` function ---")
    embed_map, decomposition, umap_model = decomposition_umap.decompose_and_embed(
        data,
        decomposition_method='cdd',
        decomposition_max_n=6,
        n_component=2,
        verbose=True
    )

    # --- Save the results from the first example to .npy files ---
    print("\n--- Saving results to .npy files ---")
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "original_data.npy"), data)
    # np.save(os.path.join(output_dir, "decomposition.npy"), decomposition)
    np.save(os.path.join(output_dir, "embed_map_x.npy"), np.array(embed_map[0]))
    np.save(os.path.join(output_dir, "embed_map_y.npy"), np.array(embed_map[1]))
    print(f"Results saved in '{output_dir}/' directory.")

    # --- Generate plots for the results ---
    print("\n--- Generating plots ---")
    plot_original_data('Input Data Components', data, pink_noise, signal_blobs)
    plot_decomposition_components('Decomposition Components via CDD', decomposition)
    plot_umap_embedding_scatter('UMAP Embedding Scatter Plot', embed_map, signal_blobs)
    plt.show()
    # --- Example 2: Using the `DecompositionUMAP` class `__init__` directly ---
    print("\n--- Running Example 2: Using the `DecompositionUMAP` class directly ---")
    decomposition_func = lambda d: decomposition_umap.multiscale_decomposition.cdd_decomposition(d, max_n=6)
    pipeline_instance = decomposition_umap.DecompositionUMAP(
        original_data=data,
        decomposition_func=decomposition_func,
        n_component=2,
        verbose=True
    )
    print("Example 2 finished. Results are numerically identical to Example 1.")

    # --- Example 3: Using the `DecompositionUMAP` class with a pre-computed decomposition ---
    print("\n--- Running Example 3: Using a pre-computed decomposition ---")
    print("Pre-computing the decomposition...")
    precomputed_decomposition, _ = decomposition_umap.multiscale_decomposition.cdd_decomposition(data, max_n=6)
    
    # ADDRESSING THE ERROR:
    # The constructor expects EITHER 'original_data' (with a function) OR a 'decomposition'.
    # Providing both triggers a ValueError. The fix is to provide only the pre-computed decomposition.
    # The class can infer the shape from decomposition.shape[1:].
    pipeline_precomputed = decomposition_umap.DecompositionUMAP(
        decomposition=precomputed_decomposition,
        n_component=2,
        verbose=True
    )
    print("Example 3 finished. Results are numerically identical to Example 1.")