import numpy as np
import pickle
from scipy.signal import hilbert
# Handle UMAP import at module level
try:
    import umap
    assert umap.UMAP() is not None
except AttributeError:
    import umap.umap_ as umap
    assert umap.UMAP() is not None



from .embedding import umap_embedding
from .utils import max_norm
from .multiscale_decomposition import (cdd_decomposition, emd_decomposition,
                                       msm_decomposition, _auto_msm_filter_sizes,
                                       adaptive_multiscale_decomposition)
class DecompositionUMAP:
    """
    Manages a decomposition-then-UMAP dimensionality reduction workflow.

    This class is the core engine that orchestrates the entire process, from
    data decomposition to UMAP model training and inference. It holds the state
    of the pipeline, including the decomposition components, the trained model,
    and all configuration parameters.

    While this class can be used directly for fine-grained control, it is often
    more convenient to use the high-level wrapper functions `decompose_and_embed`
    for training and `decompose_with_existing_model` for inference.

    Args:
        original_data (numpy.ndarray, optional): The input raw data to be
            decomposed and embedded. Required if `decomposition` is not provided.
        decomposition (numpy.ndarray, optional): A pre-computed decomposition.
            Required if `original_data` is not provided. Must have shape
            (n_components, ...original_shape).
        decomposition_func (callable, optional): A function that takes `original_data`
            and returns its decomposition. Required if `original_data` is provided.
        umap_n_neighbors (int, optional): The `n_neighbors` parameter for UMAP,
            controlling the balance between local and global structure. Defaults to 30.
        umap_min_dist (float, optional): The `min_dist` parameter for UMAP,
            controlling how tightly points are clustered. Defaults to 0.0.
        n_component (int, optional): The number of dimensions for the UMAP
            embedding. Defaults to 2.
        threshold (float, optional): A value to mask low-signal regions in the
            `original_data` before analysis. Points below this value are
            excluded. Defaults to None.
        norm_func (callable, optional): A function to normalize feature vectors
            before passing them to UMAP. If `None`, no normalization is applied.
            Defaults to None.
        use_hilbert_amplitude (bool, optional): If True, computes the amplitude of
            the analytic signal for each decomposition component before UMAP.
            Defaults to False.
        train_fraction (float, optional): Fraction of data (0.0 to 1.0) to use
            for training the UMAP model. Defaults to None (use all data).
        train_mask (numpy.ndarray, optional): A boolean mask specifying the exact
            points to use for training. Overrides `train_fraction`. Defaults to None.
        verbose (bool, optional): If True, prints progress messages. Defaults to True.

    Attributes:
        decomposition (numpy.ndarray): The decomposition components of the input data.
        original_shape (tuple): The shape of the original input data.
        embed_map (list[np.ndarray]): The resulting list of UMAP embedding maps.
        umap_model (umap.UMAP): The trained UMAP reducer instance.
        decomposition_func (callable): The function used for decomposition.
        norm_func (callable): The function used for normalization.
        threshold (float): The threshold used for masking.
        use_hilbert_amplitude (bool): Flag indicating if Hilbert amplitude was used.
        verbose (bool): Verbosity flag.

    Raises:
        ValueError: If initialization parameters are provided incorrectly (e.g.,
            both or neither of `original_data` and `decomposition` are given, or
            if `decomposition_func` is missing when needed).
    """
    def __init__(self, original_data=None, decomposition=None, decomposition_func=None,
                 umap_n_neighbors=30, umap_min_dist=0.0, n_component=2, threshold=None,
                 norm_func=None, use_hilbert_amplitude=False, train_fraction=None,
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
            self.decomposition, original_data=effective_original_data,
            umap_n_neighbors=umap_n_neighbors, umap_min_dist=umap_min_dist,
            n_component=n_component, threshold=threshold, norm_func=norm_func,
            train_fraction=train_fraction, train_mask=train_mask, verbose=verbose
        )

    def _apply_hilbert(self, decomposition_data):
        """
        Applies the Hilbert transform to each component and returns the amplitude.
        This is an internal helper method.

        Args:
            decomposition_data (numpy.ndarray): The data components.

        Returns:
            numpy.ndarray: The amplitude of the analytic signal for each component.

        Raises:
            ValueError: If the input data's shape has more than 3 dimensions.
        """
        if len(decomposition_data.shape) > 3:
            raise ValueError("Hilbert transform is only supported for 1D or 2D channel images.")
        if self.verbose: print("[Hilbert] Applying Hilbert transform and calculating amplitude.")
        return np.array([np.abs(hilbert(comp)) for comp in decomposition_data])

    def save_umap_model(self, filename):
        """
        Saves the trained UMAP model to a file using pickle.

        Args:
            filename (str): The path to the file where the model will be saved.

        Raises:
            ValueError: If the UMAP model has not been trained yet.
        """
        if self.umap_model is None: raise ValueError("UMAP model is not trained.")
        with open(filename, 'wb') as f: pickle.dump(self.umap_model, f)
        if self.verbose: print(f"[UMAP] Model saved to {filename}")

    def load_umap_model(self, filename):
        """
        Loads a UMAP model from a pickle file and sets it as the instance's model.

        Args:
            filename (str): The path to the file from which to load the model.

        Raises:
            ValueError: If the loaded object from the file is not a valid UMAP model.
            FileNotFoundError: If the filename does not exist.
        """
        with open(filename, 'rb') as f: loaded_model = pickle.load(f)
        if not isinstance(loaded_model, umap.UMAP):
            raise ValueError(f"File {filename} does not contain a valid UMAP model.")
        self.umap_model = loaded_model
        if self.verbose: print(f"[UMAP] Model loaded from {filename}")

    def compute_new_embeddings(self, new_original_data=None, new_decomposition=None):
        """
        Projects new data into the embedding space of the trained UMAP model.

        This is the primary inference method. It takes new data (either raw or
        pre-decomposed), applies the same preprocessing steps configured during
        initialization (e.g., Hilbert transform, normalization), and uses the
        trained model's `transform` method to generate the new embedding.

        Args:
            new_original_data (numpy.ndarray, optional): New raw data to decompose
                and project.
            new_decomposition (numpy.ndarray, optional): New pre-computed decomposition
                to project. Provide either this or `new_original_data`.

        Returns:
            list[np.ndarray]: A list of new embedding maps, one for each dimension,
            reshaped to match the new data's original shape.

        Raises:
            ValueError: If the UMAP model is not trained/loaded, if arguments
                are provided incorrectly, or if a decomposition function is
                unavailable when needed for `new_original_data`.
        """
        if self.umap_model is None: raise ValueError("UMAP model is not trained or loaded.")
        if not (new_original_data is None) ^ (new_decomposition is None):
            raise ValueError("Provide either 'new_original_data' or 'new_decomposition', but not both or neither.")

        final_decomposition, output_shape, effective_data = None, None, None

        if new_original_data is not None:
            if self.decomposition_func is None:
                raise ValueError("Cannot process 'new_original_data' without a 'decomposition_func'.")
            if self.verbose: print("[Inference] Decomposing new data...")
            decomp_result = self.decomposition_func(new_original_data)
            final_decomposition = np.array(decomp_result[0] if isinstance(decomp_result, tuple) else decomp_result)
            output_shape = new_original_data.shape
            effective_data = new_original_data
        else:
            if self.verbose: print("[Inference] Using new pre-computed decomposition.")
            final_decomposition = new_decomposition
            output_shape = new_decomposition.shape[1:]
            # Create a placeholder for thresholding compatibility
            effective_data = np.empty(output_shape)

        if self.use_hilbert_amplitude:
            final_decomposition = self._apply_hilbert(final_decomposition)

        if self.threshold is not None:
            if new_original_data is None:
                print("Warning: Threshold was set during training, but 'new_original_data' is not provided for inference. NaN masking will be skipped.")
            else:
                mask = np.ones(output_shape)
                mask[new_original_data < self.threshold] = np.nan
                final_decomposition = final_decomposition * mask

        # Reshape for UMAP and handle any NaN values from masking
        data_input = final_decomposition.reshape(final_decomposition.shape[0], -1)
        umap_data = data_input.T
        valid_indices = ~np.isnan(umap_data).any(axis=1)
        umap_data_filtered = umap_data[valid_indices]

        if self.norm_func is not None:
            umap_data_filtered = np.apply_along_axis(self.norm_func, 1, umap_data_filtered)

        if self.verbose: print(f"[Inference] Projecting {len(umap_data_filtered)} data points...")
        transformed_embedding = self.umap_model.transform(umap_data_filtered)

        # Reconstruct the full embedding map, inserting NaNs where data was invalid
        full_embedding = np.full((len(umap_data), self.umap_model.n_components), np.nan)
        full_embedding[valid_indices] = transformed_embedding

        return [full_embedding[:, i].reshape(output_shape) for i in range(full_embedding.shape[1])]


def decompose_with_existing_model(model_filename,
                                  data=None, datasets=None, data_multivariate=None, decomposition=None,
                                  decomposition_method='cdd', decomposition_func=None,
                                  norm_func=None, verbose=True,
                                  decomposition_max_n=None, msm_filter_sizes='auto'):
    """
    Applies a pre-trained UMAP model to new data using explicit, mutually exclusive modes.

    This is the primary high-level function for inference. It loads a saved UMAP
    model and uses it to project new data into the learned embedding space, ensuring
    consistency with the original embedding.

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
    """
    num_modes = sum(arg is not None for arg in [data, datasets, data_multivariate, decomposition])
    if num_modes != 1:
        raise ValueError("Provide exactly one of 'data', 'datasets', 'data_multivariate', or 'decomposition'.")

    # --- Load the trained model into a dummy instance ---
    temp_instance = DecompositionUMAP(decomposition=np.zeros((1, 1, 1)), verbose=False)
    temp_instance.load_umap_model(model_filename)
    temp_instance.norm_func = norm_func # Set the normalization for the transform step

    # --- Define the decomposition function if needed ---
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
    
    # --- Mode 4: Pre-computed Decomposition ---
    if decomposition is not None:
        if verbose: print("[Pre-computed Mode] Transforming the provided decomposition.")
        embed_map = temp_instance.compute_new_embeddings(new_decomposition=decomposition)
        return embed_map, decomposition

    # --- Mode 1: Single Raw Dataset ---
    elif data is not None:
        if verbose: print("[Single Dataset Mode] Decomposing and transforming data.")
        final_decomposition = _decomposition_func(data)
        embed_map = temp_instance.compute_new_embeddings(new_decomposition=final_decomposition)
        return embed_map, final_decomposition

    # --- Mode 2: Multi-Dataset (Batch) ---
    elif datasets is not None:
        if not datasets: raise ValueError("Input 'datasets' list cannot be empty.")
        if verbose: print(f"[Multi-Dataset Mode] Processing {len(datasets)} datasets...")
        decompositions = [_decomposition_func(d) for d in datasets]
        merged_decomposition = np.concatenate(decompositions, axis=1)
        merged_embed_map = temp_instance.compute_new_embeddings(new_decomposition=merged_decomposition)
        
        split_maps_by_comp = [np.split(em_comp, len(datasets), axis=0) for em_comp in merged_embed_map]
        list_of_embed_maps = [list(embeds) for embeds in zip(*split_maps_by_comp)]
        return list_of_embed_maps, decompositions

    # --- Mode 3: Multivariate Dataset ---
    elif data_multivariate is not None:
        if verbose: print(f"[Multivariate Mode] Decomposing {data_multivariate.shape[0]} channels.")
        decompositions_per_channel = [_decomposition_func(channel) for channel in data_multivariate]
        merged_decomposition = np.concatenate(decompositions_per_channel, axis=0)
        embed_map = temp_instance.compute_new_embeddings(new_decomposition=merged_decomposition)
        return embed_map, merged_decomposition


def decompose_and_embed(data=None, datasets=None, data_multivariate=None, decomposition=None,
                        decomposition_method='cdd', decomposition_func=None,
                        norm_func=None, umap_n_neighbors=30, umap_min_dist=0.0,
                        n_component=2, threshold=None, use_hilbert_amplitude=False,
                        train_fraction=None, train_mask=None, verbose=True,
                        decomposition_max_n=None, msm_filter_sizes='auto'):
    """
    Performs decomposition and UMAP embedding using explicit, mutually exclusive modes.

    This is the primary high-level function for the library. It orchestrates the
    entire training workflow, from data preparation to UMAP model creation.

    Operating Modes
    ---------------
    The function operates in one of four modes, selected by providing exactly one
    of the following keyword arguments:

    1.  **`data=...` (Single Raw Dataset)**
        -   **Input**: A single 2D NumPy array (or 1D for time series).
        -   **Behavior**: Decomposes the raw data and then runs UMAP.

    2.  **`datasets=...` (Multi-Dataset / Batch)**
        -   **Input**: A list of 2D NumPy arrays.
        -   **Behavior**: Decomposes each dataset, merges them, trains a single
            UMAP model, and splits the results back into a list.

    3.  **`data_multivariate=...` (Multivariate Raw Dataset)**
        -   **Input**: A single 3D+ NumPy array (e.g., `(channels, height, width)`).
        -   **Behavior**: Decomposes each channel and merges the components into a
            single feature set for UMAP.

    4.  **`decomposition=...` (Pre-computed Decomposition)**
        -   **Input**: A single 3D+ NumPy array of pre-computed components.
        -   **Behavior**: Skips the decomposition step entirely and proceeds
            directly to UMAP training.

    Args:
        data (numpy.ndarray, optional): Input for **Single Raw Dataset Mode**.
        datasets (list[numpy.ndarray], optional): Input for **Multi-Dataset Mode**.
        data_multivariate (numpy.ndarray, optional): Input for **Multivariate Mode**.
        decomposition (numpy.ndarray, optional): Input for **Pre-computed Mode**.
        decomposition_method (str, optional): The name of a built-in decomposition
            method. **Ignored if `decomposition` is provided**.
        decomposition_func (callable, optional): A custom decomposition function.
            **Ignored if `decomposition` is provided**.
        (Other parameters are documented in previous responses).

    Returns:
        tuple: The contents of the tuple depend on the operating mode.
    """
    num_modes = sum(arg is not None for arg in [data, datasets, data_multivariate, decomposition])
    if num_modes != 1:
        raise ValueError("Provide exactly one of 'data', 'datasets', 'data_multivariate', or 'decomposition'.")

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

    common_params = {
        'umap_n_neighbors': umap_n_neighbors, 'umap_min_dist': umap_min_dist,
        'n_component': n_component, 'threshold': threshold, 'norm_func': norm_func,
        'use_hilbert_amplitude': use_hilbert_amplitude, 'train_fraction': train_fraction,
        'train_mask': train_mask, 'verbose': verbose
    }

    # --- Mode 4: Pre-computed Decomposition ---
    if decomposition is not None:
        if verbose: print("[Pre-computed Mode] Using the provided decomposition.")
        if decomposition_method != 'cdd' or decomposition_func is not None or decomposition_max_n is not None:
            print("Warning: A pre-computed decomposition was provided. 'decomposition_method', 'decomposition_func', and 'decomposition_max_n' will be ignored.")
        instance = DecompositionUMAP(decomposition=decomposition, **common_params)
        return instance.embed_map, instance.decomposition, instance.umap_model

    # --- Mode 1: Single Raw Dataset ---
    elif data is not None:
        instance = DecompositionUMAP(
            original_data=data,
            decomposition_func=_decomposition_func,
            **common_params
        )
        return instance.embed_map, instance.decomposition, instance.umap_model

    # --- Mode 2: Multi-Dataset (Batch) ---
    elif datasets is not None:
        if not datasets: raise ValueError("Input 'datasets' list cannot be empty.")
        if verbose: print(f"[Multi-Dataset Mode] Processing {len(datasets)} datasets...")
        decompositions = [_decomposition_func(d) for d in datasets]
        merged_decomposition = np.concatenate(decompositions, axis=1)
        instance = DecompositionUMAP(decomposition=merged_decomposition, **common_params)
        merged_embed_map = instance.embed_map
        split_maps_by_comp = [np.split(em_comp, len(datasets), axis=0) for em_comp in merged_embed_map]
        list_of_embed_maps = [list(embeds) for embeds in zip(*split_maps_by_comp)]
        return list_of_embed_maps, decompositions, instance.umap_model

    # --- Mode 3: Multivariate Dataset ---
    elif data_multivariate is not None:
        if verbose: print(f"[Multivariate Mode] Decomposing {data_multivariate.shape[0]} channels.")
        decompositions_per_channel = [_decomposition_func(channel) for channel in data_multivariate]
        merged_decomposition = np.concatenate(decompositions_per_channel, axis=0)
        instance = DecompositionUMAP(decomposition=merged_decomposition, **common_params)
        return instance.embed_map, instance.decomposition, instance.umap_model