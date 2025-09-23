import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys

# --- Add parent directory to path to find the 'src' package ---
# This is a common pattern to make a local package (in a 'src' folder)
# importable when the script is run from a different directory (like 'tests' or 'examples').
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Import the main package ---
import src as decomposition_umap

# --- Import a specific decomposition function for the pre-computed example ---
# This is needed for Example 3 where we run the decomposition manually first.
from src.multiscale_decomposition import cdd_decomposition

# ==============================================================================
# --- 1. Data Generation Functions ---
# These functions create a synthetic dataset for testing the UMAP workflow.
# The goal is to create a known signal (Gaussian blobs) hidden in noise.
# ==============================================================================

def generate_pink_noise(shape):
    """Generates pink noise with a 1/f power spectrum."""
    rows, cols = shape
    # Create a frequency grid
    u, v = np.fft.fftfreq(rows), np.fft.fftfreq(cols)
    frequency_radius = np.sqrt(u[:, np.newaxis]**2 + v**2)
    frequency_radius[0, 0] = 1.0  # Avoid division by zero at the DC component
    
    # Generate white noise in the frequency domain and scale it by 1/f
    fft_white_noise = np.fft.fft2(np.random.randn(rows, cols))
    fft_pink_noise = fft_white_noise / frequency_radius
    
    # Transform back to the spatial domain and normalize
    pink_noise = np.real(np.fft.ifft2(fft_pink_noise))
    
    return (pink_noise - pink_noise.mean()) / pink_noise.std()

def add_gaussian_blobs(data, centers, sigmas, amplitudes):
    """Adds Gaussian blobs (the "signal") to an existing data array (the "noise")."""
    rows, cols = data.shape
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Create a separate array to hold only the signal for later comparison
    signal = np.zeros_like(data, dtype=float)
    for center, sigma, amp in zip(centers, sigmas, amplitudes):
        cx, cy, sx, sy = *center, *sigma
        signal += amp * np.exp(-(((x - cx)**2 / (2 * sx**2)) + ((y - cy)**2 / (2 * sy**2))))
        
    # Return the combined data and the signal-only mask
    return data + signal, signal

# ==============================================================================
# --- 2. Plotting Functions ---
# These functions visualize the input, intermediate steps, and final output
# of the Decomposition-UMAP process.
# ==============================================================================

def plot_original_data(title, data, noise, signal):
    """Generates a figure showing the noise, the signal, and the combined data."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    im1 = axes[0].imshow(noise, cmap='gray', origin='lower'); axes[0].set_title('Pink Noise Background'); fig.colorbar(im1, ax=axes[0])
    im2 = axes[1].imshow(signal, cmap='gray', origin='lower'); axes[1].set_title('Gaussian Blobs (Signal)'); fig.colorbar(im2, ax=axes[1])
    im3 = axes[2].imshow(data, cmap='viridis', origin='lower'); axes[2].set_title('Final Data (Noise + Signal)'); fig.colorbar(im3, ax=axes[2])
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

def plot_decomposition_components(title, decomposition):
    """Generates a figure showing all decomposition components in a grid."""
    num_components = decomposition.shape[0]
    # Calculate a pleasing grid shape for the subplots
    ncols = int(np.ceil(np.sqrt(num_components)))
    nrows = int(np.ceil(num_components / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = axes.flatten() # Flatten the 2D array of axes for easy iteration
    
    # Plot each component
    for i in range(num_components):
        im = axes[i].imshow(decomposition[i], cmap='viridis', origin='lower'); axes[i].set_title(f'Component {i+1}'); fig.colorbar(im, ax=axes[i])
    
    # Hide any unused subplot axes if the number of components is not a perfect square
    for j in range(num_components, len(axes)):
        axes[j].axis('off')
        
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

# MODIFIED: Function now accepts pink_noise and a highlight_factor
def plot_umap_embedding_scatter(title, embed_map, signal_blobs, pink_noise, highlight_factor=1.0):
    """
    Generates a scatter plot of the UMAP embedding, highlighting significant signal points.
    
    Args:
        title (str): The main title for the plot.
        embed_map (list): The list of UMAP component maps.
        signal_blobs (np.ndarray): The array containing only the signal values.
        pink_noise (np.ndarray): The array containing only the noise values.
        highlight_factor (float): A multiplier to define "significant" signal.
            A point is highlighted if its signal value is greater than its noise
            value multiplied by this factor.
    """
    # Flatten the 2D embedding maps into 1D arrays for plotting
    umap_x, umap_y = embed_map[0].flatten(), embed_map[1].flatten()
    
    # MODIFIED: The new logic for identifying significant signal points.
    # A point is now considered "signal" only if its Gaussian blob value is
    # greater than the underlying pink noise value times the factor.
    is_signal = signal_blobs.flatten() > (pink_noise.flatten() * highlight_factor)
    
    # Separate the data into signal and noise sets
    noise_x, noise_y = umap_x[~is_signal], umap_y[~is_signal]
    signal_x, signal_y = umap_x[is_signal], umap_y[is_signal]
    
    plt.figure(figsize=(8, 8))
    # Plot the noise points first with high transparency to show density
    plt.scatter(noise_x, noise_y, label='Noise / Weak Signal', alpha=0.1, s=10, color='gray')
    # Plot the significant signal points on top so they are clearly visible
    plt.scatter(signal_x, signal_y, label=f'Significant Signal (Signal > Noise * {highlight_factor})', alpha=0.8, s=15, color='red')
    
    plt.title(title, fontsize=16)
    plt.xlabel('UMAP Dimension 1'); plt.ylabel('UMAP Dimension 2')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.6); plt.axis('equal')
    plt.tight_layout()

# ==============================================================================
# --- 3. Main Execution Block ---
# This block demonstrates the various ways to use the decomposition_umap library.
# ==============================================================================

if __name__ == '__main__':
    # --- Configuration ---
    # Define a factor to determine what counts as a "significant" signal for plotting.
    # A factor of 2.0 means we only highlight points where the signal is at least
    # twice as strong as the background noise.
    HIGHLIGHT_FACTOR = 2.0

    # --- Generate a common dataset for all examples ---
    print("--- Generating synthetic data: Gaussian blobs in pink noise ---")
    shape = (256, 256)
    pink_noise = generate_pink_noise(shape)
    data, signal_blobs = add_gaussian_blobs(
        pink_noise,
        centers=[(60, 80), (160, 180), (100, 200)],
        sigmas=[(5, 5), (2, 8), (2, 2)],
        amplitudes=[5.0, 4.0, 3.0]
    )

    # --- Example 1: High-Level API `decompose_and_embed` ---
    # This is the simplest and most common way to use the library for training.
    print("\n--- Running Example 1: High-level `decompose_and_embed` function ---")
    embed_map, decomposition, umap_model = decomposition_umap.decompose_and_embed(
        data=data,
        decomposition_method='cdd',
        decomposition_max_n=6,
        n_component=2,
        verbose=True
    )

    # --- Save the results from the first example ---
    print("\n--- Saving results to .npy files ---")
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "original_data.npy"), data)
    np.save(os.path.join(output_dir, "decomposition.npy"), decomposition)
    np.save(os.path.join(output_dir, "embed_map_x.npy"), np.array(embed_map[0]))
    np.save(os.path.join(output_dir, "embed_map_y.npy"), np.array(embed_map[1]))
    print(f"Results saved in '{output_dir}/' directory.")

    # --- Generate plots for the results from Example 1 ---
    print("\n--- Generating plots ---")
    plot_original_data('Input Data Components', data, pink_noise, signal_blobs)
    plot_decomposition_components('Decomposition Components via CDD', decomposition)
    # MODIFIED: Pass the pink_noise array and the factor to the plotting function
    plot_umap_embedding_scatter(
        'UMAP Embedding with Significant Signal Highlighted',
        embed_map,
        signal_blobs,
        pink_noise,
        highlight_factor=HIGHLIGHT_FACTOR
    )
    
    # --- Example 2: Using the `DecompositionUMAP` Class Directly ---
    # This approach is useful if you want more control.
    print("\n--- Running Example 2: Using the `DecompositionUMAP` class directly ---")
    pipeline_instance = decomposition_umap.DecompositionUMAP(
        original_data=data,
        decomposition_method='cdd',
        decomposition_max_n=6,
        n_component=2,
        verbose=True
    )
    print("Example 2 finished. Results are numerically identical to Example 1.")

    # --- Example 3: Using a Pre-computed Decomposition with the Class ---
    # This is efficient if your decomposition is slow.
    print("\n--- Running Example 3: Using the `DecompositionUMAP` class with a pre-computed decomposition ---")
    print("Pre-computing the decomposition...")
    precomputed_decomposition, _ = cdd_decomposition(data, max_n=6)
    
    pipeline_precomputed = decomposition_umap.DecompositionUMAP(
        decomposition=np.array(precomputed_decomposition),
        n_component=2,
        verbose=True
    )
    print("Example 3 finished. Results are numerically identical to Example 1.")
    
    # --- Display all created plot windows at the end of the script ---
    print("\n--- Displaying all plots. ---")
    plt.show()