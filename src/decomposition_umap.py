import numpy as np
import pickle
from scipy.signal import hilbert

# --- Placeholder Imports for Decomposition Functions ---
# In a real application, these would point to the actual library functions.
from .multiscale_decomposition import (cdd_decomposition, emd_decomposition, 
                                       msm_decomposition, _auto_msm_filter_sizes, 
                                       adaptive_multiscale_decomposition)
# Handle UMAP import at module level
try:
    import umap
    print(umap.UMAP())
except AttributeError:
    import umap.umap_ as umap


def max_norm(vector):
    """
    Normalizes a vector by its maximum absolute value.
    Args:
        vector (numpy.ndarray): Input vector to normalize.
    Returns:
        numpy.ndarray: Normalized vector.
    """
    max_val = np.max(np.abs(vector))
    return vector / max_val if max_val > 0 else vector


def umap_embedding(decomposition, original_data, n_component=2, umap_n_neighbors=30, 
                   umap_min_dist=0.0, threshold=None, norm_func=max_norm, 
                   train_fraction=None, train_mask=None, verbose=True):
    """
    Perform UMAP dimensionality reduction on decomposed data, optionally training on a subset.
    This is a standalone utility function that can be used directly.
    """
    original_shape = original_data.shape
    
    # Apply thresholding if specified, creating a NaN mask
    if threshold is not None:
        mask = np.ones(original_shape)
        mask[original_data < threshold] = np.nan
        # Element-wise multiplication will propagate NaNs
        decomposition = decomposition * mask.reshape(1, -1) if mask.ndim == 1 else decomposition * mask

    # Reshape decomposition for UMAP: (n_samples, n_features)
    data_input = decomposition.reshape(decomposition.shape[0], -1)
    
    # Filter out any data points (columns) that contain NaNs
    valid_column_indices = np.where(np.all(~np.isnan(data_input), axis=0))[0]
    filtered_data = data_input[:, valid_column_indices]
    umap_data = filtered_data.T # Shape is now (n_valid_points, n_components)

    # Determine which data points to use for training the UMAP model
    if train_mask is not None:
        if train_mask.shape != original_shape:
            raise ValueError(f"train_mask shape {train_mask.shape} must match original_data shape {original_shape}")
        train_mask_flat = train_mask.flatten()[valid_column_indices]
        train_indices = np.where(train_mask_flat)[0]
    elif train_fraction is not None:
        if not (0 < train_fraction <= 1):
            raise ValueError("train_fraction must be between 0 and 1")
        n_samples = umap_data.shape[0]
        n_train = int(n_samples * train_fraction)
        train_indices = np.random.choice(n_samples, size=n_train, replace=False)
    else:
        train_indices = np.arange(umap_data.shape[0])

    if verbose:
        print(f"[UMAP] Training on {len(train_indices)} of {umap_data.shape[0]} valid data points.")

    umap_data_train = umap_data[train_indices]

    if norm_func is not None:
        umap_data_train = np.apply_along_axis(norm_func, 1, umap_data_train)

    # Initialize and fit the UMAP model
    reducer = umap.UMAP(n_components=n_component, n_neighbors=umap_n_neighbors, 
                        min_dist=umap_min_dist, verbose=verbose)
    umap_model = reducer.fit(umap_data_train)

    # If training was on a subset, transform the entire valid dataset.
    if train_mask is not None or train_fraction is not None:
        if verbose: print("[UMAP] Transforming the full dataset with the trained model.")
        umap_data_full = umap_data
        if norm_func is not None:
            umap_data_full = np.apply_along_axis(norm_func, 1, umap_data_full)
        transformed_embedding = umap_model.transform(umap_data_full)
    else:
        transformed_embedding = umap_model.embedding_

    # Reconstruct the full embedding map, inserting NaNs where data was invalid
    full_embedding = np.full((np.prod(original_shape), n_component), np.nan)
    full_embedding[valid_column_indices] = transformed_embedding
    
    embed_map = [full_embedding[:, i].reshape(original_shape) for i in range(n_component)]

    return embed_map, umap_model


class DecompositionUMAP:
    """
    A flexible class to manage decomposition and UMAP dimensionality reduction.
    """
    def __init__(self, original_data=None, decomposition=None, decomposition_func=None, 
                 umap_n_neighbors=30, umap_min_dist=0.0, n_component=2, threshold=None, 
                 norm_func=max_norm, use_hilbert_amplitude=False, train_fraction=None, 
                 train_mask=None, verbose=True):
        
        if not (original_data is None) ^ (decomposition is None):
            raise ValueError("Provide either 'original_data' (with 'decomposition_func') or a 'decomposition', but not both or neither.")

        if original_data is not None and not callable(decomposition_func):
            raise ValueError("If 'original_data' is provided, a callable 'decomposition_func' is required.")

        self.decomposition_func = decomposition_func
        self.norm_func = norm_func
        self.use_hilbert_amplitude = use_hilbert_amplitude
        self.threshold = threshold
        self.verbose = verbose
        
        effective_original_data = None

        if original_data is not None:
            if verbose: print("[Init] Decomposing original data...")
            decomp_result = self.decomposition_func(original_data)
            self.decomposition = np.array(decomp_result[0] if isinstance(decomp_result, tuple) else decomp_result)
            self.original_shape = original_data.shape
            effective_original_data = original_data
        else:
            if verbose: print("[Init] Using pre-computed decomposition.")
            self.decomposition = decomposition
            self.original_shape = decomposition.shape[1:]
            effective_original_data = np.empty(self.original_shape)
        
        if use_hilbert_amplitude:
            self.decomposition = self._apply_hilbert(self.decomposition)

        if verbose: print("[Init] Training UMAP model...")
        self.embed_map, self.umap_model = umap_embedding(
            self.decomposition,
            original_data=effective_original_data,
            umap_n_neighbors=umap_n_neighbors,
            umap_min_dist=umap_min_dist,
            n_component=n_component,
            threshold=threshold,
            norm_func=norm_func,
            train_fraction=train_fraction,
            train_mask=train_mask,
            verbose=verbose
        )

    def _apply_hilbert(self, decomposition_data):
        if len(decomposition_data.shape) > 3:
            raise ValueError("Hilbert transform is only supported for 1D or 2D channel images.")
        if self.verbose: print("[Hilbert] Applying Hilbert transform and calculating amplitude.")
        return np.array([np.abs(hilbert(comp)) for comp in decomposition_data])

    def save_umap_model(self, filename):
        if self.umap_model is None: raise ValueError("UMAP model is not trained.")
        with open(filename, 'wb') as f: pickle.dump(self.umap_model, f)
        if self.verbose: print(f"[UMAP] Model saved to {filename}")

    def load_umap_model(self, filename):
        with open(filename, 'rb') as f: loaded_model = pickle.load(f)
        if not isinstance(loaded_model, umap.UMAP): raise ValueError(f"File {filename} does not contain a valid UMAP model.")
        self.umap_model = loaded_model
        if self.verbose: print(f"[UMAP] Model loaded from {filename}")

    def compute_new_embeddings(self, new_original_data=None, new_decomposition=None):
        if self.umap_model is None: raise ValueError("UMAP model is not trained or loaded.")
        if not (new_original_data is None) ^ (new_decomposition is None):
            raise ValueError("Provide either 'new_original_data' or 'new_decomposition', but not both or neither.")

        final_decomposition, output_shape = None, None
        effective_original_data_for_threshold = None

        if new_original_data is not None:
            if self.decomposition_func is None:
                raise ValueError("Cannot process 'new_original_data' because the class was not initialized with a 'decomposition_func'.")
            if self.verbose: print("[Inference] Decomposing new data...")
            decomp_result = self.decomposition_func(new_original_data)
            final_decomposition = np.array(decomp_result[0] if isinstance(decomp_result, tuple) else decomp_result)
            output_shape = new_original_data.shape
            effective_original_data_for_threshold = new_original_data
        else:
            if self.verbose: print("[Inference] Using new pre-computed decomposition.")
            final_decomposition = new_decomposition
            output_shape = new_decomposition.shape[1:]
            effective_original_data_for_threshold = np.empty(output_shape)

        if self.use_hilbert_amplitude:
            final_decomposition = self._apply_hilbert(final_decomposition)

        if self.threshold is not None:
            if new_original_data is None:
                print("Warning: Threshold was set during training, but 'new_original_data' is not provided for inference. NaN handling will be skipped.")
            else:
                mask = np.ones(output_shape)
                mask[new_original_data < self.threshold] = np.nan
                final_decomposition = final_decomposition * mask

        data_input = final_decomposition.reshape(final_decomposition.shape[0], -1)
        umap_data = data_input.T
        
        valid_indices = ~np.isnan(umap_data).any(axis=1)
        umap_data_filtered = umap_data[valid_indices]
        
        if self.norm_func is not None:
            umap_data_filtered = np.apply_along_axis(self.norm_func, 1, umap_data_filtered)
        
        if self.verbose: print(f"[Inference] Projecting {len(umap_data_filtered)} data points...")
        transformed_embedding = self.umap_model.transform(umap_data_filtered)
        
        full_embedding = np.full((len(umap_data), self.umap_model.n_components), np.nan)
        full_embedding[valid_indices] = transformed_embedding
        
        new_embed_map = [full_embedding[:, i].reshape(output_shape) for i in range(full_embedding.shape[1])]
        return new_embed_map


def decompose_and_embed(data, decomposition_method='cdd', decomposition_func=None, norm_func=max_norm, **kwargs):
    """High-level wrapper to perform decomposition and UMAP embedding in one step."""
    
    if decomposition_func is None:
        def func(d):
            if decomposition_method == 'cdd': return cdd_decomposition(d, max_n=kwargs.get('cdd_max_n'))
            elif decomposition_method == 'emd': return emd_decomposition(d, max_imf=kwargs.get('emd_max_imf', -1))
            elif decomposition_method == 'amd': return adaptive_multiscale_decomposition(d, max_n=kwargs.get('cdd_max_n'))
            elif decomposition_method == 'msm':
                msm_sizes = kwargs.get('msm_filter_sizes', 'auto')
                if msm_sizes == 'auto': msm_sizes = _auto_msm_filter_sizes(d.shape)
                return msm_decomposition(d, *msm_sizes)
            else: raise ValueError(f"Unknown decomposition_method: {decomposition_method}")
        decomposition_func = func

    instance = DecompositionUMAP(original_data=data, decomposition_func=decomposition_func, norm_func=norm_func, **kwargs)
    return instance.embed_map, instance.decomposition, instance.umap_model


def decompose_with_existing_model(model_filename, data=None, decomposition=None, decomposition_method='cdd',
                                 decomposition_func=None, norm_func=max_norm, **kwargs):
    """Apply a pre-trained UMAP model to new data or decomposition."""
    
    # Create a dummy instance to load the model into
    temp_instance = DecompositionUMAP(decomposition=np.zeros((1, 1)), verbose=False) # Minimal valid init
    temp_instance.load_umap_model(model_filename)
    
    # Store settings from the loaded model if needed, though UMAP doesn't expose all of them easily.
    # For this implementation, we assume the user provides consistent settings (like norm_func).
    temp_instance.norm_func = norm_func
    
    if decomposition_func is None and data is not None:
        def func(d):
            if decomposition_method == 'cdd': return cdd_decomposition(d, max_n=kwargs.get('cdd_max_n'))
            elif decomposition_method == 'emd': return emd_decomposition(d, max_imf=kwargs.get('emd_max_imf', -1))
            # ... add other methods as needed
            else: raise ValueError(f"Unknown decomposition_method: {decomposition_method}")
        temp_instance.decomposition_func = func
    else:
        temp_instance.decomposition_func = decomposition_func

    # Use the instance's compute method
    embed_map = temp_instance.compute_new_embeddings(new_original_data=data, new_decomposition=decomposition)
    
    # To return the decomposition, we must compute it if it wasn't provided
    if decomposition is None:
        decomp_result = temp_instance.decomposition_func(data)
        final_decomposition = np.array(decomp_result[0] if isinstance(decomp_result, tuple) else decomp_result)
    else:
        final_decomposition = decomposition

    return embed_map, final_decomposition