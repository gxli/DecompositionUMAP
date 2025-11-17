import numpy as np
import argparse
import sys
import os
import json

# --- Import the main library ---
# This relative import works because __main__.py is inside the package
from . import decomposition_umap

# --- Check for astropy for FITS support ---
try:
    from astropy.io import fits
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False

def load_data(file_path):
    """Loads data from either a .npy or .fits file."""
    file_path = os.path.normpath(file_path)
    if not os.path.exists(file_path):
        print(f"Error: Input file not found at '{file_path}'")
        sys.exit(1)

    if file_path.lower().endswith('.npy'):
        try:
            return np.load(file_path)
        except Exception as e:
            print(f"Error: Failed to load .npy file. {e}"); sys.exit(1)
            
    elif file_path.lower().endswith('.fits'):
        if not ASTROPY_AVAILABLE:
            print("Error: 'astropy' is required for .fits files. Install it with: pip install astropy"); sys.exit(1)
        try:
            with fits.open(file_path) as hdul:
                return hdul[0].data.astype(np.float32)
        except Exception as e:
            print(f"Error: Failed to load .fits file. {e}"); sys.exit(1)
    else:
        print("Error: Unsupported file format. Please provide a .npy or .fits file."); sys.exit(1)

def main():
    """Main function to run the command-line interface."""
    # --- Setup Command-Line Argument Parsing with improved help ---
    parser = argparse.ArgumentParser(
        prog="decomp-umap",
        description="A command-line tool to process FITS and NPY files with Decomposition-UMAP.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Usage Examples:
-----------------
1. Basic analysis (saves results next to the input file):
   decomp-umap path/to/my_image.fits

2. Specify an output directory and apply a log10 transform and a threshold:
   decomp-umap path/to/my_image.fits -o results/ -t 0.5 --log_base 10

3. 3D embedding with a specific decomposition level:
   decomp-umap my_data.npy -o results/ -d 8 -n 3
"""
    )
    parser.add_argument("input_file", type=str, help="Path to the input .fits or .npy file.")
    parser.add_argument("-o", "--output-dir", type=str, default=None,
                        help="Optional: Directory to save the output files.\n"
                             "If not provided, files are saved in the same directory as the input file.")
    parser.add_argument("-t", "--threshold", type=float, default=None,
                        help="Optional: A lower threshold value. Data points in the input file\n"
                             "below this value will be excluded from the analysis.")
    parser.add_argument("-d", "--decomposition-level", type=int, default=None,
                        help="Optional: The number of decomposition components (max_n).\n"
                             "If not provided, a default is calculated based on image size.")
    
    # --- NEW ARGUMENT FOR LOG BASE ---
    parser.add_argument(
        "--log_base",
        type=float,
        default=None,
        help="Optional: Apply a logarithm of this base to the data *before* decomposition.\n"
             "For example, use 10 for log10 or 2.718 for natural log."
    )
    
    parser.add_argument("-n", "--n_components", type=int, default=2, choices=[2, 3],
                        help="The number of UMAP embedding dimensions (default: 2).")
    parser.add_argument("-m", "--decomposition-method", type=str, default="cdd", choices=["cdd", "emd"],
                        help="The decomposition method to use (default: 'cdd').")
    parser.add_argument("-p", "--umap_params", type=str, default=None,
                        help="Optional: A JSON string of advanced parameters for the UMAP constructor.\n"
                             "Example: '{\"n_neighbors\": 50, \"min_dist\": 0.0, \"low_memory\": true}'")
    parser.add_argument("--no-verbose", action="store_true", help="Run in quiet mode.")
    
    args = parser.parse_args()

    # --- 1. Load Data ---
    print(f"--- Loading data from '{args.input_file}' ---")
    data = load_data(args.input_file)
    print(f"Successfully loaded data with shape: {data.shape}")

    # --- 2. Determine Decomposition Level ---
    if args.decomposition_level is None:
        min_dim = min(data.shape)
        if min_dim < 8: cdd_max_n = 2
        else: cdd_max_n = max(1, int(np.log2(min_dim) - 2))
        print(f"No decomposition level provided. Using default -> cdd_max_n = {cdd_max_n}")
    else:
        cdd_max_n = args.decomposition_level
        print(f"Using user-provided decomposition level -> cdd_max_n = {cdd_max_n}")

    # --- 3. Parse Advanced UMAP Parameters ---
    umap_params = {}
    if args.umap_params:
        try:
            umap_params = json.loads(args.umap_params)
            print(f"Using advanced UMAP parameters: {umap_params}")
        except json.JSONDecodeError:
            print("Error: Invalid JSON string for --umap_params."); sys.exit(1)

    # --- 4. Run Main Library Function ---
    print("\n--- Running decomposition and embedding... ---")
        
    embed_map, decomposition, _ = decomposition_umap.decompose_and_embed(
        data=data,
        decomposition_method=args.decomposition_method,
        decomposition_max_n=cdd_max_n,
        decomposition_log_base=args.log_base,
        n_component=args.n_components,
        umap_params=umap_params,
        threshold=args.threshold,
        verbose=not args.no_verbose
    )
    print("Processing complete.")

    # --- 5. Determine Output Path and Save Results ---
    if args.output_dir is None:
        output_directory = os.path.dirname(args.input_file)
    else:
        output_directory = args.output_dir
        os.makedirs(output_directory, exist_ok=True)
    
    print(f"\n--- Saving results to '{os.path.abspath(output_directory)}/' ---")
    base_name = os.path.splitext(os.path.basename(args.input_file))[0]
    
    decomp_path = os.path.join(output_directory, f"{base_name}_decomposition.npy")
    np.save(decomp_path, decomposition)
    print(f"Saved decomposition to: {decomp_path}")
    
    for i in range(args.n_components):
        comp_name = ['x', 'y', 'z'][i]
        umap_path = os.path.join(output_directory, f"{base_name}_umap_{comp_name}.npy")
        np.save(umap_path, embed_map[i])
        print(f"Saved UMAP component '{comp_name}' to: {umap_path}")

    print("\n--- Command-line tool finished successfully. ---")

if __name__ == '__main__':
    main()