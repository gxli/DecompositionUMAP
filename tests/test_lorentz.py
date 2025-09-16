# examples/run_lorenz_example.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sys
import os

# --- Import the required decomposition library ---
import constrained_diffusion as cdd

# --- Add parent directory to path to find the 'src' package ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Import your package ---
import src as dumap


def generate_lorenz_timeseries(n_points=100000):
    """Generates a 1D time series of a specified length using the Lorenz system."""
    print(f"Generating {n_points} data points from the Lorenz system...")
    def lorenz_system(t, xyz, sigma, rho, beta):
        x, y, z = xyz
        dxdt = sigma * (y - x); dydt = x * (rho - z) - y; dzdt = x * y - beta * z
        return [dxdt, dydt, dzdt]
    sigma, rho, beta = 10, 28, 8/3; initial_state = [0., 1., 1.05]
    dt = 0.01; t_max = dt * n_points; t_eval = np.arange(0, t_max, dt)
    solution = solve_ivp(lorenz_system, [0, t_max], initial_state, args=(sigma, rho, beta), dense_output=True, t_eval=t_eval)
    return solution.y[0]

def run_lorenz_cdd_umap_example():
    """
    Runs the full workflow and returns the original data and UMAP embedding.
    
    Returns:
        tuple: A tuple containing:
            - lorenz_data (numpy.ndarray): The original 100,000-point time series.
            - umap_x (numpy.ndarray): The x-component of the UMAP embedding.
            - umap_y (numpy.ndarray): The y-component of the UMAP embedding.
    """
    N_TOTAL_POINTS = 10000; TRAIN_FRACTION = 1
    N_EMBED_POINTS_TO_PLOT = 10000; CDD_NUM_COMPONENTS = 12

    lorenz_data = generate_lorenz_timeseries(n_points=N_TOTAL_POINTS)
    
    print("\nDecomposing the time series using CDD...")
    components, _ = cdd.constrained_diffusion_decomposition(lorenz_data, num_channels=CDD_NUM_COMPONENTS)
    decomposition = np.array(components)
    print(f"Shape of CDD decomposition: {decomposition.shape}")

    print(f"\nTraining UMAP model using {TRAIN_FRACTION * 100}% of the data...")
    decomp_umap_instance = dumap.DecompositionUMAP(
        decomposition=decomposition,
        train_fraction=TRAIN_FRACTION,
        n_component=2,
        umap_n_neighbors=50,
        umap_min_dist=0.0,
        verbose=True
    )
    
    # Extract the full embedding for all 100,000 points
    umap_x = decomp_umap_instance.embed_map[0]
    umap_y = decomp_umap_instance.embed_map[1]
    
    # --- Plotting Section ---
    print(f"\nPlotting the first {N_EMBED_POINTS_TO_PLOT} points of the embedding...")
    
    # We only need to slice the arrays for plotting, not for returning
    points_to_plot = np.vstack([umap_x[:N_EMBED_POINTS_TO_PLOT], umap_y[:N_EMBED_POINTS_TO_PLOT]]).T
    
    plt.figure(figsize=(12, 10))
    time_colors = np.arange(N_EMBED_POINTS_TO_PLOT)
    scatter = plt.scatter(points_to_plot[:, 0], points_to_plot[:, 1], s=10, c=time_colors, cmap='plasma')
    plt.title(f"UMAP Embedding of Lorenz Attractor (Decomposed with CDD)", fontsize=16)
    plt.xlabel("UMAP Component 1", fontsize=12); plt.ylabel("UMAP Component 2", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    cbar = plt.colorbar(scatter); cbar.set_label("Time Step Index", fontsize=12)
    plt.show()

    # --- Return the full data arrays ---
    return lorenz_data, umap_x, umap_y


if __name__ == "__main__":
    # Run the main function and capture the results
    lorenz_data, umap_x, umap_y = run_lorenz_cdd_umap_example()

    # --- Save the returned results to .npy files ---
    OUTPUT_DIR = "results"
    print(f"\nSaving results to the '{OUTPUT_DIR}' directory...")
    
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Define file paths
    data_path = os.path.join(OUTPUT_DIR, 'lorenz_data.npy')
    umap_x_path = os.path.join(OUTPUT_DIR, 'umap_x.npy')
    umap_y_path = os.path.join(OUTPUT_DIR, 'umap_y.npy')
    
    # Save the arrays
    np.save(data_path, lorenz_data)
    np.save(umap_x_path, umap_x)
    np.save(umap_y_path, umap_y)
    
    print(f"Successfully saved:\n- {data_path}\n- {umap_x_path}\n- {umap_y_path}")