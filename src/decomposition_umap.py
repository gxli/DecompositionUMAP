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

def get_file_extension(file_path):
    """Extracts the lowercase extension from a file path."""
    return os.path.splitext(file_path)[1].lower()

def load_data(file_path):
    """Loads data from either a .npy or .fits file."""
    file_path = os.path.normpath(file_path)
    if not os.path.exists(file_path):
        print(f"Error: Input file not found at '{file_path}'")
        sys.exit(1)

    ext = get_file_extension(file_path)

    if ext == '.npy':
        try:
            return np.load(file_path)
        except Exception as e:
            print(f"Error: Failed to load .npy file. {e}"); sys.exit(1)
            
    elif ext == '.fits':
        if not ASTROPY_AVAILABLE:
            print("Error: 'astropy' is required for .fits files. Install it with: pip install astropy"); sys.exit(1)
        try:
            with fits.open(file_path) as hdul:
                # Assuming data is in the primary HDU (index 0)
                return hdul[0].data.astype(np.float32)
        except Exception as e:
            print(f"Error: Failed to load .fits file. {e}"); sys.exit(1)
    else:
        print("Error: Unsupported file format. Please provide a .npy or .fits file."); sys.exit(1)

def save_data(file_path, data, fmt_extension):
    """Saves data to .npy or .fits based on the provided extension."""
    if fmt_extension == '.fits':
        if not ASTROPY_AVAILABLE:
            print("Error: Cannot save to .fits because 'astropy' is not installed."); sys.exit(1)
        try:
            # Create a Primary HDU
            hdu = fits.PrimaryHDU(data)
            # Overwrite if exists
            hdu.writeto(file_path, overwrite=True)
            print(f"Saved: {file_path}")
        except Exception as e:
            print(f"Error: Failed to save .fits file to '{file_path}'. {e}"); sys.exit(1)
    else:
        # Default to .npy logic
        if not file_path.endswith('.npy'):
            file_path += '.npy'
        try:
            np.save(file_path, data)
            print(f"Saved: {file_path}")
        except Exception as e:
            print(f"Error: Failed to save .npy file to '{file_path}'. {e}"); sys.exit(1)

def main():
    """Main function to run the command-line interface."""
    # --- Setup Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(
        prog="decomp-umap",
        description="A command-line tool to process FITS and NPY files with Decomposition-UMAP.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Usage Examples:
-----------------
1. Basic analysis (outputs .fits if input is .fits):
   decomp-umap path/to/image.fits

2. Use a pre-calculated decomposition file (skip internal decomposition step):
   decomp-umap path/to/image.fits --load-decomposition path/to/existing_decomp.fits

3. Specify output directory and specific parameters:
   decomp-umap data.npy -o results/ -t 0.5 --log_base 10
"""
    )
    parser.add_argument("input_file", type=str, help="Path to the input .fits or .npy file (Raw Data).")
    
    parser.add_argument("--load-decomposition", type=str, default=None,
                        help="Optional: Path to a pre-computed decomposition file (.fits or .npy).\n"
                             "If provided, the decomposition step is skipped, and this data is used for embedding.")

    parser.add_argument("-o", "--output-dir", type=str, default=None,
                        help="Optional: Directory to save the output files.\n"
                             "If not provided, files are saved in the same directory as the input file.")
    parser.add_argument("-t", "--threshold", type=float, default=None,
                        help="Optional: A lower threshold value. Data points in the input file\n"
                             "below this value will be excluded from the analysis.")
    parser.add_argument("-d", "--decomposition-level", type=int, default=None,
                        help="Optional: The number of decomposition components (max_n).\n"
                             "If not provided, a default is calculated based on image size.")
    
    parser.add_argument("--log_base", type=float, default=None,
                        help="Optional: Apply a logarithm of this base to the data *before* decomposition.\n"
                             "Example: 10 for log10.")
    
    parser.add_argument("-n", "--n_components", type=int, default=2, choices=[2, 3],
                        help="The number of UMAP embedding dimensions (default: 2).")
    parser.add_argument("-m", "--decomposition-method", type=str, default="cdd", choices=["cdd", "emd", "wavelet"],
                        help="The decomposition method to use (default: 'cdd').")
    parser.add_argument("-p", "--umap_params", type=str, default=None,
                        help="Optional: A JSON string of advanced parameters for the UMAP constructor.\n"
                             "Example: '{\"n_neighbors\": 50, \"min_dist\": 0.0, \"low_memory\": true}'")
    parser.add_argument("--no-verbose", action="store_true", help="Run in quiet mode.")
    
    args = parser.parse_args()
    verbose = not args.no_verbose

    # --- 1. Load Raw Data ---
    if verbose: print(f"--- Loading raw data from '{args.input_file}' ---")
    raw_data = load_data(args.input_file)
    input_ext = get_file_extension(args.input_file)
    if verbose: print(f"Successfully loaded data with shape: {raw_data.shape}")

    # --- 2. Load Optional Pre-computed Decomposition ---
    pre_computed_decomp = None
    if args.load_decomposition:
        if verbose: print(f"--- Loading pre-computed decomposition from '{args.load_decomposition}' ---")
        pre_computed_decomp = load_data(args.load_decomposition)
        if verbose: print(f"Loaded decomposition with shape: {pre_computed_decomp.shape}")

    # --- 3. Determine Decomposition Level (if needed) ---
    # Only strictly necessary if we aren't providing a pre-computed decomposition, 
    # but useful for logging.
    cdd_max_n = args.decomposition_level
    if cdd_max_n is None and pre_computed_decomp is None:
        min_dim = min(raw_data.shape)
        if min_dim < 8: cdd_max_n = 2
        else: cdd_max_n = max(1, int(np.log2(min_dim) - 2))
        if verbose: print(f"No decomposition level provided. Using default -> cdd_max_n = {cdd_max_n}")

    # --- 4. Parse Advanced UMAP Parameters ---
    umap_params = {}
    if args.umap_params:
        try:
            umap_params = json.loads(args.umap_params)
            if verbose: print(f"Using advanced UMAP parameters: {umap_params}")
        except json.JSONDecodeError:
            print("Error: Invalid JSON string for --umap_params."); sys.exit(1)

    # --- 5. Run Main Library Function ---
    print("\n--- Running decomposition and embedding... ---")
    
    # If pre_computed_decomp is provided, we pass it to 'decomposition'.
    # 'data' is still passed as raw_data to allow the library to handle thresholding if needed.
    
    try:
        embed_map, final_decomposition, _ = decomposition_umap.decompose_and_embed(
            data=raw_data if pre_computed_decomp is None else None,
            decomposition=pre_computed_decomp,
            decomposition_method=args.decomposition_method,
            decomposition_max_n=cdd_max_n,
            decomposition_log_base=args.log_base,
            n_component=args.n_components,
            umap_params=umap_params,
            threshold=args.threshold,
            verbose=verbose
        )
    except Exception as e:
        print(f"Error during execution: {e}")
        sys.exit(1)

    print("Processing complete.")

    # --- 6. Determine Output Path and Save Results ---
    if args.output_dir is None:
        output_directory = os.path.dirname(args.input_file)
    else:
        output_directory = args.output_dir
        os.makedirs(output_directory, exist_ok=True)
    
    print(f"\n--- Saving results to '{os.path.abspath(output_directory)}/' ---")
    base_name = os.path.splitext(os.path.basename(args.input_file))[0]
    
    # Save Decomposition
    # We preserve the input extension (e.g., if input was .fits, save decomp as .fits)
    decomp_filename = f"{base_name}_decomposition{input_ext}"
    decomp_path = os.path.join(output_directory, decomp_filename)
    save_data(decomp_path, final_decomposition, input_ext)
    
    # Save Embedding Components
    for i in range(args.n_components):
        # Naming convention: ...dim1.fits, ...dim2.fits
        comp_filename = f"{base_name}_dim{i+1}{input_ext}"
        umap_path = os.path.join(output_directory, comp_filename)
        save_data(umap_path, embed_map[i], input_ext)

    print("\n--- Command-line tool finished successfully. ---")

if __name__ == '__main__':
    main()