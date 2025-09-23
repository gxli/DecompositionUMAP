import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# Import for the custom decomposition function example
from scipy.ndimage import gaussian_filter

# --- Add parent directory to path to find the 'src' package ---
# This is a common pattern to make a local package (in a 'src' folder)
# importable when the script is run from a different directory (like 'tests' or 'examples').
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Import the main package ---
import src as decomposition_umap

# --- Import the example data generator and a specific decomposition function ---
from src import example as du_example
from src.multiscale_decomposition import cdd_decomposition

# ==============================================================================
# --- 1. Custom Decomposition Function ---
# This function is kept locally in the test script to demonstrate how a user
# would define and pass their own function to the library.
# ==============================================================================

def custom_gaussian_decomposition(data):
    """
    A simple custom decomposition function that separates data into two scales
    using Gaussian filtering. This demonstrates the `decomposition_func` argument.
    """
    # Create a blurred component (large scale features)
    blurred_component = gaussian_filter(data, sigma=5)
    # Create a detail component (small scale features)
    detail_component = data - blurred_component
    # Stack them into the required shape: (n_components, height, width)
    return np.array([blurred_component, detail_component])

# ==============================================================================
# --- 2. Plotting Functions ---
# These functions visualize the input, intermediate steps, and final output.
# ==============================================================================

def plot_original_data(title, data, background, feature):
    """Generates a figure showing the background, the feature/anomaly, and the combined data."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    im1 = axes[0].imshow(background, cmap='gray', origin='lower'); axes[0].set_title('Background Signal'); fig.colorbar(im1, ax=axes[0])
    im2 = axes[1].imshow(feature, cmap='gray', origin='lower'); axes[1].set_title('Anomaly / Feature'); fig.colorbar(im2, ax=axes[1])
    im3 = axes[2].imshow(data, cmap='viridis', origin='lower'); axes[2].set_title('Final Combined Data'); fig.colorbar(im3, ax=axes[2])
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
        im = axes[i].imshow(decomposition[i], cmap='viridis', origin='lower'); axes[i].set_title(f'Component {i+1}'); fig.colorbar(im, ax=axes[i])
    for j in range(num_components, len(axes)):
        axes[j].axis('off')
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

# MODIFIED: Function now accepts the combined 'data' array for the new condition
def plot_umap_embedding_scatter(title, embed_map, data, anomaly_mask, f_anomaly=0.1):
    """
    Generates a scatter plot of the UMAP embedding, highlighting anomaly points
    based on the condition that the anomaly value is greater than the total data value.
    """
    umap_x, umap_y = embed_map[0].flatten(), embed_map[1].flatten()
    
    # MODIFIED: The new logic for identifying significant anomaly points.
    # A point is highlighted only if its anomaly component value is strictly
    # greater than the total data value (signal + anomaly) at that point.
    # This implies the background signal must be negative.
    is_highlighted = anomaly_mask.flatten() > data.flatten() * f_anomaly
    
    # Separate the data into background and highlighted sets
    background_x, background_y = umap_x[~is_highlighted], umap_y[~is_highlighted]
    highlight_x, highlight_y = umap_x[is_highlighted], umap_y[is_highlighted]
    
    plt.figure(figsize=(8, 8))
    # Plot the background points with high transparency
    plt.scatter(background_x, background_y, label='Background', alpha=0.1, s=10, color='gray')
    # Plot the highlighted points on top so they are clearly visible
    plt.scatter(highlight_x, highlight_y, label='Highlighted (Anomaly > Data)', alpha=0.8, s=15, color='red')
    
    plt.title(title, fontsize=16)
    plt.xlabel('UMAP Dimension 1'); plt.ylabel('UMAP Dimension 2')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.6); plt.axis('equal')
    plt.tight_layout()

# ==============================================================================
# --- 3. Main Execution Block ---
# This block demonstrates the various ways to use the decomposition_umap library.
# ==============================================================================

if __name__ == '__main__':
    # --- Generate a common dataset for all examples ---
    print("--- Generating synthetic data: Gaussian anomaly in fractal noise ---")
    data, signal, anomaly = du_example.generate_fractal_with_gaussian()

    # --- Example 1: Standard Mode with a Built-in Decomposition Method ---
    # This is the simplest and most common way to use the library for training.
    print("\n--- Running Example 1: Standard Mode (`data` and `decomposition_method`) ---")
    embed_map, decomposition, umap_model = decomposition_umap.decompose_and_embed(
        data=data,
        decomposition_method='cdd',
        decomposition_max_n=6,
        n_component=2,
        umap_params={'n_neighbors': 20, 'min_dist': 0.0},
        verbose=True
    )

    # --- Save and plot the results from Example 1 ---
    print("\n--- Saving results from Example 1 ---")
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "fractal_data.npy"), data)
    np.save(os.path.join(output_dir, "fractal_decomposition.npy"), decomposition)
    np.save(os.path.join(output_dir, "fractal_embed_map_x.npy"), embed_map[0])
    np.save(os.path.join(output_dir, "fractal_embed_map_y.npy"), embed_map[1])
    print(f"Results saved in '{output_dir}/'")

    print("\n--- Generating plots for Example 1 ---")
    plot_original_data('Input Data for All Examples', data, signal, anomaly)
    plot_decomposition_components('Decomposition from Example 1', decomposition)
    # MODIFIED: Pass the combined 'data' array to the plotting function
    plot_umap_embedding_scatter(
        'UMAP Embedding with Anomaly > Data Highlighted',
        embed_map,
        data, # Pass the combined data for the new condition
        anomaly
    )

    # --- Example 2: Using a Custom Decomposition Function ---
    # This approach is for when you have your own specific method for separating features.
    print("\n--- Running Example 2: Custom Decomposition Function Mode (`decomposition_func=...`) ---")
    embed_map_custom, decomp_custom, _ = decomposition_umap.decompose_and_embed(
        data=data,
        decomposition_func=custom_gaussian_decomposition,
        n_component=2,
        verbose=True
    )
    print("Example 2 finished. A plot for its custom decomposition could also be generated.")

    # --- Example 3: Using a Pre-computed Decomposition ---
    # This is efficient if your decomposition is slow and you want to reuse it
    # while experimenting with different UMAP parameters.
    print("\n--- Running Example 3: Pre-computed Decomposition Mode (`decomposition=...`) ---")
    # Step 1: Manually compute the decomposition first
    print("Step 1: Pre-computing decomposition...")
    precomputed_decomposition, _ = cdd_decomposition(data, max_n=6)
    
    # Step 2: Pass the result directly to the function
    print("\nStep 2: Passing pre-computed decomposition to the wrapper...")
    embed_map_pre, _, _ = decomposition_umap.decompose_and_embed(
        decomposition=np.array(precomputed_decomposition),
        n_component=2,
        verbose=True,
    )
    print("Example 3 finished successfully.")
    
    # --- Display all created plot windows at the end of the script ---
    print("\n--- Displaying all plots. ---")
    plt.show()