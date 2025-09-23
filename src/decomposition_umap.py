import numpy as np
import pickle
from scipy.signal import hilbert

# --- Robust UMAP import with assertion ---
try:
    import umap
    # Assert that the UMAP class itself exists
    assert umap.UMAP
except (ImportError, AttributeError, AssertionError):
    # Fallback for older versions or different package structures
    try:
        import umap.umap_ as umap
        assert umap.UMAP
    except (ImportError, AssertionError):
        # If both attempts fail, raise a clear error message
        raise ImportError(
            "UMAP class not found. Please ensure umap-learn is installed correctly. "
            "You can install it with: pip install umap-learn"
        )

from .embedding import umap_embedding
from .utils import max_norm
from .multiscale_decomposition import (cdd_decomposition, emd_decomposition)

class DecompositionUMAP:
    """
    Manages a decomposition-then-UMAP dimensionality reduction workflow.

    This class is the core engine that orchestrates the entire process, from
    data decomposition to UMAP model training and inference. It holds the state
    of the pipeline, including the decomposition components, the trained model,
    and all configuration parameters.

    There are two primary ways to initialize this class:
    1.  **With Raw Data**: Provide `original_data`. You can either provide a specific
        `decomposition_func` or use the `decomposition_method` and
        `decomposition_max_n` keywords to have the class build one for you.
    2.  **With a Pre-computed Decomposition**: Provide a `decomposition`. The class
        will skip the decomposition step and proceed directly to UMAP training.

    Args:
        original_data (numpy.ndarray, optional): The input raw data. If provided,
            a decomposition function will be run on it.
        decomposition (numpy.ndarray, optional): A pre-computed decomposition.
            This takes precedence over `original_data` if both are provided.
        decomposition_func (callable, optional): A custom function that takes `original_data`
            and returns its decomposition. This takes precedence over `decomposition_method`.
        decomposition_method (str, optional): The name of a built-in decomposition
            method (e.g., 'cdd', 'emd') to use if `original_data` is provided
            without a `decomposition_func`. Defaults to 'cdd'.
        decomposition_max_n (int, optional): The number of components to generate for
            the selected `decomposition_method`. Defaults to None.
        umap_n_neighbors (int, optional): The `n_neighbors` parameter for UMAP.
            Defaults to 15. This is a convenience parameter; for more control,
            use `umap_params`.
        umap_min_dist (float, optional): The `min_dist` parameter for UMAP.
            Defaults to 0.1. This is a convenience parameter.
        low_memory (bool, optional): Sets the `low_memory` option in UMAP, which
            trades speed for reduced memory usage. Defaults to False.
        n_component (int, optional): The number of dimensions for the UMAP embedding.
        threshold (float, optional): A value to mask low-signal regions. Requires
            `original_data` to be effective.
        norm_func (callable, optional): A function to normalize feature vectors.
        use_hilbert_amplitude (bool, optional): If True, computes the analytic signal amplitude.
        train_fraction (float, optional): Fraction of data to use for UMAP training.
        train_mask (numpy.ndarray, optional): A boolean mask specifying training points.
        verbose (bool, optional): If True, prints progress messages.
        umap_params (dict, optional): For advanced use. A dictionary of keyword
            arguments passed directly to the `umap.UMAP` constructor. If provided,
            it will override settings like `umap_n_neighbors` and `low_memory`.
            Example: `{'metric': 'cosine'}`.

    Attributes:
        decomposition (numpy.ndarray): The decomposition components of the input data.
        original_shape (tuple): The shape of the original input data.
        embed_map (list[np.ndarray]): The resulting list of UMAP embedding maps.
        umap_model (umap.UMAP): The trained UMAP reducer instance.
        decomposition_func (callable): The function used for decomposition. It will
            be populated even if `decomposition_method` is used for initialization.
    """
    def __init__(self, original_data=None, decomposition=None, decomposition_func=None,
                 decomposition_method='cdd', decomposition_max_n=None,
                 umap_n_neighbors=15, umap_min_dist=0.1, low_memory=False,
                 n_component=2, threshold=None, norm_func=None,
                 use_hilbert_amplitude=False, train_fraction=None,
                 train_mask=None, verbose=True, umap_params=None):

        final_decomp_func = decomposition_func

        if decomposition is not None:
            if verbose: print("[Init] Using pre-computed decomposition.")
            self.decomposition = np.array(decomposition)
            if original_data is None:
                if threshold is not None:
                    print("Warning: A 'threshold' was provided without 'original_data'. Thresholding will be skipped.")
                self.original_shape = self.decomposition.shape[1:]
                effective_original_data = np.empty(self.original_shape)
            else:
                self.original_shape = original_data.shape
                effective_original_data = original_data
        
        elif original_data is not None:
            if final_decomp_func is None:
                if verbose: print(f"[Init] Building decomposition function from method: '{decomposition_method}'")
                def func(d):
                    if decomposition_method == 'cdd':
                        res = cdd_decomposition(d, max_n=decomposition_max_n)
                    elif decomposition_method == 'emd':
                        max_imf = decomposition_max_n if decomposition_max_n is not None else -1
                        res = emd_decomposition(d, max_imf=max_imf)
                    else:
                        raise ValueError(f"Unknown decomposition_method: {decomposition_method}")
                    return np.array(res[0] if isinstance(res, tuple) else res)
                final_decomp_func = func

            if not callable(final_decomp_func):
                raise ValueError("Could not find or build a valid decomposition function.")

            if verbose: print("[Init] Decomposing original data...")
            self.decomposition = final_decomp_func(original_data)
            self.original_shape = original_data.shape
            effective_original_data = original_data
        else:
            raise ValueError("Provide either a 'decomposition' or 'original_data'.")
        
        self.decomposition_func = final_decomp_func
        self.norm_func = norm_func
        self.use_hilbert_amplitude = use_hilbert_amplitude
        self.threshold = threshold
        self.verbose = verbose

        # --- Intelligent Merging of UMAP Parameters ---
        final_umap_params = {
            'n_neighbors': umap_n_neighbors,
            'min_dist': umap_min_dist,
            'low_memory': low_memory
        }
        if umap_params:
            final_umap_params.update(umap_params)

        if use_hilbert_amplitude:
            self.decomposition = self._apply_hilbert(self.decomposition)

        if verbose: print("[Init] Training UMAP model...")
        self.embed_map, self.umap_model = umap_embedding(
            self.decomposition,
            original_data=effective_original_data,
            n_component=n_component,
            threshold=threshold,
            norm_func=norm_func,
            train_fraction=train_fraction,
            train_mask=train_mask,
            umap_params=final_umap_params,
            verbose=verbose
        )

    def _apply_hilbert(self, decomposition_data):
        """Applies the Hilbert transform and returns the amplitude."""
        if len(decomposition_data.shape) > 3:
            raise ValueError("Hilbert transform is only supported for 1D or 2D channel images.")
        if self.verbose: print("[Hilbert] Applying Hilbert transform and calculating amplitude.")
        return np.array([np.abs(hilbert(comp)) for comp in decomposition_data])

    def save_umap_model(self, filename):
        """Saves the trained UMAP model to a file using pickle."""
        if self.umap_model is None: raise ValueError("UMAP model is not trained.")
        with open(filename, 'wb') as f: pickle.dump(self.umap_model, f)
        if self.verbose: print(f"[UMAP] Model saved to {filename}")

    def load_umap_model(self, filename):
        """Loads a UMAP model from a pickle file."""
        with open(filename, 'rb') as f: loaded_model = pickle.load(f)
        if not isinstance(loaded_model, umap.UMAP):
            raise ValueError(f"File {filename} does not contain a valid UMAP model.")
        self.umap_model = loaded_model
        if self.verbose: print(f"[UMAP] Model loaded from {filename}")

    def compute_new_embeddings(self, new_original_data=None, new_decomposition=None):
        """Projects new data into the embedding space of the trained UMAP model."""
        if self.umap_model is None: raise ValueError("UMAP model is not trained or loaded.")
        if not (new_original_data is None) ^ (new_decomposition is None):
            raise ValueError("Provide either 'new_original_data' or 'new_decomposition', but not both or neither.")

        if new_original_data is not None:
            if self.decomposition_func is None:
                raise ValueError("Cannot process 'new_original_data' because class was not initialized with a decomposition_func.")
            if self.verbose: print("[Inference] Decomposing new data...")
            final_decomposition = self.decomposition_func(new_original_data)
        else:
            if self.verbose: print("[Inference] Using new pre-computed decomposition.")
            final_decomposition = np.array(new_decomposition)

        output_shape = final_decomposition.shape[1:]

        if self.use_hilbert_amplitude:
            final_decomposition = self._apply_hilbert(final_decomposition)

        if self.threshold is not None:
            if new_original_data is None:
                print("Warning: Thresholding skipped for inference without 'new_original_data'.")
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

        return [full_embedding[:, i].reshape(output_shape) for i in range(full_embedding.shape[1])]


def decompose_and_embed(data=None, datasets=None, data_multivariate=None, decomposition=None,
                        decomposition_method='cdd', decomposition_func=None,
                        norm_func=None, umap_n_neighbors=15, umap_min_dist=0.1, low_memory=False,
                        n_component=2, threshold=None, use_hilbert_amplitude=False,
                        train_fraction=None, train_mask=None, verbose=True,
                        decomposition_max_n=None, umap_params=None):
    """
    Performs decomposition and UMAP embedding using explicit, mutually exclusive modes.

    This is the primary high-level function for the library, orchestrating the
    entire training workflow from data preparation to UMAP model creation.

    Operating Modes
    ---------------
    The function operates in one of four modes, selected by providing exactly one
    of the following keyword arguments: `data`, `datasets`, `data_multivariate`, or `decomposition`.

    Args:
        data (numpy.ndarray, optional): For **Single Raw Dataset Mode**.
        datasets (list[numpy.ndarray], optional): For **Multi-Dataset (Batch) Mode**.
        data_multivariate (numpy.ndarray, optional): For **Multivariate Mode**.
        decomposition (numpy.ndarray, optional): For **Pre-computed Decomposition Mode**.
        decomposition_method (str, optional): The name of the decomposition method (e.g., 'cdd').
        decomposition_max_n (int, optional): Number of components for the decomposition.
        umap_n_neighbors (int, optional): The `n_neighbors` parameter for UMAP. Defaults to 15.
        umap_min_dist (float, optional): The `min_dist` parameter for UMAP. Defaults to 0.1.
        low_memory (bool, optional): Sets the `low_memory` option in UMAP. Defaults to False.
        n_component (int, optional): The target dimension for the UMAP embedding.
        umap_params (dict, optional): For advanced use. A dictionary of keyword
            arguments passed directly to `umap.UMAP`. Overrides other UMAP-specific
            arguments like `umap_n_neighbors` if there is a conflict.
        (Other parameters are documented in previous responses).

    Returns:
        tuple: The results corresponding to the operating mode.
    """
    num_modes = sum(arg is not None for arg in [data, datasets, data_multivariate, decomposition])
    if num_modes != 1:
        raise ValueError("Provide exactly one of 'data', 'datasets', 'data_multivariate', or 'decomposition'.")

    common_params = {
        'n_component': n_component, 'threshold': threshold, 'norm_func': norm_func,
        'use_hilbert_amplitude': use_hilbert_amplitude, 'train_fraction': train_fraction,
        'train_mask': train_mask, 'verbose': verbose,
        'umap_n_neighbors': umap_n_neighbors,
        'umap_min_dist': umap_min_dist,
        'low_memory': low_memory,
        'umap_params': umap_params
    }
    
    decomp_init_params = {
        'decomposition_method': decomposition_method,
        'decomposition_max_n': decomposition_max_n
    }

    if decomposition is not None:
        instance = DecompositionUMAP(decomposition=decomposition, **common_params)
        return instance.embed_map, instance.decomposition, instance.umap_model
    elif data is not None:
        instance = DecompositionUMAP(original_data=data, decomposition_func=decomposition_func, **decomp_init_params, **common_params)
        return instance.embed_map, instance.decomposition, instance.umap_model
    elif datasets is not None:
        _decomposition_func = decomposition_func
        if _decomposition_func is None:
            def func(d):
                res = cdd_decomposition(d, max_n=decomposition_max_n)
                return np.array(res[0] if isinstance(res, tuple) else res)
            _decomposition_func = func
        decompositions = [_decomposition_func(d) for d in datasets]
        merged_decomposition = np.concatenate(decompositions, axis=1)
        instance = DecompositionUMAP(decomposition=merged_decomposition, **common_params)
        merged_embed_map = instance.embed_map
        split_maps_by_comp = [np.split(em_comp, len(datasets), axis=0) for em_comp in merged_embed_map]
        list_of_embed_maps = [list(embeds) for embeds in zip(*split_maps_by_comp)]
        return list_of_embed_maps, decompositions, instance.umap_model
    elif data_multivariate is not None:
        _decomposition_func = decomposition_func
        if _decomposition_func is None:
            def func(d):
                res = cdd_decomposition(d, max_n=decomposition_max_n)
                return np.array(res[0] if isinstance(res, tuple) else res)
            _decomposition_func = func
        decompositions_per_channel = [_decomposition_func(channel) for channel in data_multivariate]
        merged_decomposition = np.concatenate(decompositions_per_channel, axis=0)
        instance = DecompositionUMAP(decomposition=merged_decomposition, original_data=data_multivariate[0], **common_params)
        return instance.embed_map, instance.decomposition, instance.umap_model


def decompose_with_existing_model(model_filename,
                                  data=None, datasets=None, data_multivariate=None, decomposition=None,
                                  decomposition_method='cdd', decomposition_func=None,
                                  norm_func=None, verbose=True,
                                  decomposition_max_n=None):
    """
    Applies a pre-trained UMAP model to new data using explicit, mutually exclusive modes.

    This is the primary high-level function for inference. It loads a saved UMAP
    model and uses it to project new data into the learned embedding space, ensuring
    consistency with the original embedding. UMAP-specific training parameters
    (like n_neighbors) are not needed as they are inherent to the loaded model.

    Operating Modes
    ---------------
    The function operates in one of four modes, selected by providing exactly one
    of the following keyword arguments:

    1.  **`data=...` (Single Raw Dataset)**
        -   **Input**: A single 2D NumPy array (or 1D time series).
        -   **Behavior**: Decomposes the raw data and transforms it using the loaded model.

    2.  **`datasets=...` (Multi-Dataset / Batch)**
        -   **Input**: A list of 2D NumPy arrays.
        -   **Behavior**: Decomposes each dataset, merges them, transforms the combined
            data with the loaded model, and splits the result back into a list.

    3.  **`data_multivariate=...` (Multivariate Raw Dataset)**
        -   **Input**: A single 3D+ NumPy array (e.g., `(channels, height, width)`).
        -   **Behavior**: Decomposes each channel, merges the components, and transforms
            the combined feature set with the loaded model.

    4.  **`decomposition=...` (Pre-computed Decomposition)**
        -   **Input**: A single 3D+ NumPy array of pre-computed components.
        -   **Behavior**: Skips decomposition and directly transforms the provided data.

    Args:
        model_filename (str): The path to the saved (pickled) UMAP model file.
        data (numpy.ndarray, optional): Input for **Single Raw Dataset Mode**.
        datasets (list[numpy.ndarray], optional): Input for **Multi-Dataset Mode**.
        data_multivariate (numpy.ndarray, optional): Input for **Multivariate Mode**.
        decomposition (numpy.ndarray, optional): Input for **Pre-computed Mode**.
        decomposition_method (str, optional): The name of the decomposition method.
            **Must be consistent** with the method used to train the model.
        decomposition_func (callable, optional): A custom decomposition function.
            **Must be consistent** with the one used during training.
        norm_func (callable, optional): The normalization function. **Must be consistent**
            with the one used during training. Defaults to None.
        verbose (bool, optional): If True, prints progress messages.
        decomposition_max_n (int, optional): Number of decomposition components.
            **Must be consistent** with the value used during training.

    Returns:
        tuple: The contents of the tuple depend on the operating mode:
        - For `data` or `decomposition` mode: `(embed_map, final_decomposition)`
        - For `datasets` mode: `(list_of_embed_maps, list_of_decompositions)`
        - For `data_multivariate` mode: `(embed_map, merged_decomposition)`

    Raises:
        ValueError: If more or less than one of `data`, `datasets`,
            `data_multivariate`, or `decomposition` are provided.
        FileNotFoundError: If the `model_filename` is not found.
    """
    num_modes = sum(arg is not None for arg in [data, datasets, data_multivariate, decomposition])
    if num_modes != 1:
        raise ValueError("Provide exactly one of 'data', 'datasets', 'data_multivariate', or 'decomposition'.")

    temp_instance = DecompositionUMAP(decomposition=np.zeros((1, 1, 1)), verbose=False)
    temp_instance.load_umap_model(model_filename)
    temp_instance.norm_func = norm_func

    _decomposition_func = decomposition_func
    if _decomposition_func is None:
        def func(d):
            if decomposition_method == 'cdd':
                res = cdd_decomposition(d, max_n=decomposition_max_n)
            elif decomposition_method == 'emd':
                max_imf = decomposition_max_n if decomposition_max_n is not None else -1
                res = emd_decomposition(d, max_imf=max_imf)
            else:
                raise ValueError(f"Unknown decomposition_method: {decomposition_method}")
            return np.array(res[0] if isinstance(res, tuple) else res)
        _decomposition_func = func
    
    if decomposition is not None:
        embed_map = temp_instance.compute_new_embeddings(new_decomposition=decomposition)
        return embed_map, decomposition
    elif data is not None:
        final_decomposition = _decomposition_func(data)
        embed_map = temp_instance.compute_new_embeddings(new_decomposition=final_decomposition, new_original_data=data)
        return embed_map, final_decomposition
    elif datasets is not None:
        decompositions = [_decomposition_func(d) for d in datasets]
        merged_decomposition = np.concatenate(decompositions, axis=1)
        merged_embed_map = temp_instance.compute_new_embeddings(new_decomposition=merged_decomposition)
        split_maps_by_comp = [np.split(em_comp, len(datasets), axis=0) for em_comp in merged_embed_map]
        list_of_embed_maps = [list(embeds) for embeds in zip(*split_maps_by_comp)]
        return list_of_embed_maps, decompositions
    elif data_multivariate is not None:
        decompositions_per_channel = [_decomposition_func(channel) for channel in data_multivariate]
        merged_decomposition = np.concatenate(decompositions_per_channel, axis=0)
        embed_map = temp_instance.compute_new_embeddings(new_decomposition=merged_decomposition, new_original_data=data_multivariate[0])
        return embed_map, merged_decomposition