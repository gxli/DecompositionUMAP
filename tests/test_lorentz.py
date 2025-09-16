# examples/run_lorenz_example.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

# --- Import the required decomposition library ---
import constrained_diffusion as cdd

# --- Add parent directory to path to find the 'src' package ---
import sys
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
    Runs the full workflow and returns the data, UMAP embedding, the trained
    class instance, and the full decomposition.
    """
    N_TOTAL_POINTS = 10000; TRAIN_FRACTION = 1
    N_EMBED_POINTS_TO_PLOT = 2000; CDD_NUM_COMPONENTS = 12

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
        verbose=True,
        norm_func=None
    )
    
    umap_x = decomp_umap_instance.embed_map[0]
    umap_y = decomp_umap_instance.embed_map[1]
    
    # This initial plot is now optional, as the final plot is more comprehensive.
    # You can uncomment it if you still want to see the initial plot.
    # print(f"\nPlotting the first {N_EMBED_POINTS_TO_PLOT} points of the embedding...")
    # points_to_plot = np.vstack([umap_x[:N_EMBED_POINTS_TO_PLOT], umap_y[:N_EMBED_POINTS_TO_PLOT]]).T
    # plt.figure(figsize=(12, 10))
    # time_colors = np.arange(N_EMBED_POINTS_TO_PLOT)
    # plt.scatter(points_to_plot[:, 0], points_to_plot[:, 1], s=10, c=time_colors, cmap='plasma')
    # plt.title(f"UMAP Embedding of Lorenz Attractor (Initial Training)", fontsize=16)
    # plt.xlabel("UMAP Component 1"); plt.ylabel("UMAP Component 2")
    # plt.grid(True, linestyle='--', alpha=0.6)
    # cbar = plt.colorbar(scatter); cbar.set_label("Time Step Index")
    # plt.show()

    return lorenz_data, umap_x, umap_y, decomp_umap_instance, decomposition


if __name__ == "__main__":
    lorenz_data, umap_x, umap_y, decomp_umap_instance, decomposition = run_lorenz_cdd_umap_example()

    OUTPUT_DIR = "results"
    print(f"\nSaving initial results to the '{OUTPUT_DIR}' directory...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    np.save(os.path.join(OUTPUT_DIR, 'lorenz_data.npy'), lorenz_data)
    np.save(os.path.join(OUTPUT_DIR, 'umap_x.npy'), umap_x)
    np.save(os.path.join(OUTPUT_DIR, 'umap_y.npy'), umap_y)
    
    print(f"Successfully saved initial results.")

    # --- TEST compute_new_embeddings on a new slice of data ---
    N_TEST_POINTS = 2000
    print(f"\n--- Testing compute_new_embeddings on the LAST {N_TEST_POINTS} points ---")
    
    decomposition_to_embed = decomposition[:, -N_TEST_POINTS:]
    
    new_embed_map = decomp_umap_instance.compute_new_embeddings(
        new_decomposition=decomposition_to_embed
    )

    new_embed_x = new_embed_map[0]
    new_embed_y = new_embed_map[1]
    
    print(f"\nSaving new embedding results with prefix 'new_embed_'...")
    np.save(os.path.join(OUTPUT_DIR, 'new_embed_umap_x.npy'), new_embed_x)
    np.save(os.path.join(OUTPUT_DIR, 'new_embed_umap_y.npy'), new_embed_y)
    print(f"Successfully saved new embedding results.")

    # --- PLOT BOTH EMBEDDINGS FOR VISUAL CONFIRMATION ---
    print("\nPlotting original and new embeddings for visual confirmation...")
    plt.figure(figsize=(12, 10))
    
    # 1. Plot the full original embedding using a colormap for time
    time_colors = np.arange(len(umap_x))
    plt.scatter(
        umap_x, 
        umap_y, 
        s=5, 
        c=time_colors, 
        cmap='cividis', 
        alpha=0.5, 
        label='Original Full Embedding'
    )
    
    # 2. Plot the newly projected points on top with a distinct color
    new_points_to_plot = np.vstack([new_embed_x, new_embed_y]).T
    plt.scatter(
        new_points_to_plot[:, 0], 
        new_points_to_plot[:, 1], 
        s=20, 
        c='red', 
        edgecolor='black',
        linewidth=0.5,
        label=f'Newly Embedded Points (Last {N_TEST_POINTS})'
    )
    
    plt.title("compute_new_embeddings Test: New vs. Original Embedding", fontsize=16)
    plt.xlabel("UMAP Component 1", fontsize=12)
    plt.ylabel("UMAP Component 2", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    cbar = plt.colorbar()
    cbar.set_label("Time Step Index (for Original Embedding)")
    plt.show()