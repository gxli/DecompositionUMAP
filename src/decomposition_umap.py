import numpy as np
import pickle
from scipy.signal import hilbert
from .multiscale_decomposition import cdd_decomposition, emd_decomposition, msm_decomposition, _auto_msm_filter_sizes, adaptive_multiscale_decomposition

# Handle UMAP import at module level
try:
    import umap
    print(umap.UMAP())
except AttributeError:
    import umap.umap_ as umap


def max_norm(vector):
    """Normalize a vector by its maximum absolute value.

    Args:
        vector (numpy.ndarray): Input vector to normalize.

    Returns:
        numpy.ndarray: Normalized vector.
    """
    max_norm_value = np.max(np.abs(vector))
    if max_norm_value == 0:
        return vector
    return vector / max_norm_value


def umap_embedding(decomposition, original_data, n_component=2, umap_n_neighbors=30, umap_min_dist=0.0, threshold=None, norm_func=None, train_fraction=None, train_mask=None, verbose=True):
    """Perform UMAP dimensionality reduction on decomposed data, optionally training on a subset.

    Args:
        decomposition (numpy.ndarray): Decomposed data from CDD, EMD, MSM, AMD, or custom function.
        original_data (numpy.ndarray): Original input data for thresholding and reshaping.
        n_component (int, optional): Number of UMAP components. Defaults to 2.
        umap_n_neighbors (int, optional): Number of neighbors for UMAP. Defaults to 30.
        umap_min_dist (float, optional): Minimum distance in UMAP embedding. Defaults to 0.0.
        threshold (float, optional): Threshold to mask data below this value with NaN. Defaults to None.
        norm_func (callable, optional): Normalization function applied to rows. Defaults to None.
        train_fraction (float, optional): Fraction of data to use for training UMAP (0 to 1). Defaults to None.
        train_mask (numpy.ndarray, optional): Boolean mask to select training data. Defaults to None.
        verbose (bool, optional): Whether to enable verbose output. Defaults to True.

    Returns:
        list: List of arrays, each a UMAP component reshaped to the original data shape.
        umap.UMAP: Trained UMAP model.
    """
    if threshold is not None:
        mask = np.ones_like(original_data)
        mask[original_data < threshold] = np.nan
        decomposition = decomposition * mask

    original_shape = original_data.shape
    first_dim = decomposition.shape[0]
    second_dim = np.prod(original_shape)
    data_input = decomposition.reshape(first_dim, second_dim)

    valid_column_indices = np.where(np.all(~np.isnan(data_input), axis=0))[0]
    filtered_data = data_input[:, valid_column_indices]

    umap_data = filtered_data.T

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
        print(f"[UMAP] Selected {len(train_indices)} samples for training")

    umap_data_train = umap_data[train_indices]

    if norm_func is None:
        umap_data_normed = umap_data_train
    else:
        umap_data_normed = np.apply_along_axis(norm_func, 1, umap_data_train)

    if verbose:
        print("[UMAP] Starting processing")
    
    reducer = umap.UMAP(n_components=n_component, n_neighbors=umap_n_neighbors, min_dist=umap_min_dist, verbose=verbose)
    umap_model = reducer.fit(umap_data_normed)

    if train_mask is not None or train_fraction is not None:
        if norm_func is not None:
            umap_data_full = np.apply_along_axis(norm_func, 1, umap_data)
        else:
            umap_data_full = umap_data
        embedding = umap_model.transform(umap_data_full)
    else:
        embedding = umap_model.embedding_

    embed_map = [embedding[:, i].reshape(original_shape) for i in range(embedding.shape[1])]

    return embed_map, umap_model


class DecompositionUMAP:
    """A class to perform UMAP dimensionality reduction on pre-computed decomposition.

    Args:
        decomposition (numpy.ndarray): Pre-computed decomposition (e.g., from CDD, EMD, MSM, AMD, or custom function).
        original_data (numpy.ndarray): Original input data for thresholding and reshaping.
        decomposition_func (callable, optional): Function to compute decomposition for new data. Must return a numpy.ndarray or a tuple where the first element is the decomposition. Defaults to None.
        umap_n_neighbors (int, optional): Number of neighbors for UMAP. Defaults to 30.
        umap_min_dist (float, optional): Minimum distance in UMAP embedding. Defaults to 0.0.
        n_component (int, optional): Number of UMAP components. Defaults to 2.
        threshold (float, optional): Threshold to mask data below this value with NaN. Defaults to None.
        norm_func (callable, optional): Normalization function applied to rows. Defaults to None.
        use_hilbert_amplitude (bool, optional): If True, applies Hilbert transform to get the analytical amplitude of each decomposition component. Defaults to False.
        train_fraction (float, optional): Fraction of data to use for training UMAP (0 to 1). Defaults to None.
        train_mask (numpy.ndarray, optional): Boolean mask to select training data. Defaults to None.
        verbose (bool, optional): Whether to enable verbose output. Defaults to True.
    """
    def __init__(self, decomposition, original_data, decomposition_func=None, umap_n_neighbors=30, umap_min_dist=0.0, n_component=2,
                 threshold=None, norm_func=None, use_hilbert_amplitude=False, train_fraction=None, train_mask=None, verbose=True):
        if decomposition is None:
            raise ValueError("Decomposition must be provided")
        if original_data is None:
            raise ValueError("Original data must be provided")
        if train_mask is not None and train_fraction is not None:
            raise ValueError("Only one of train_mask or train_fraction can be provided")
        if decomposition_func is not None and not callable(decomposition_func):
            raise ValueError("decomposition_func must be a callable function")

        self.decomposition = decomposition
        self.decomposition_func = decomposition_func
        self.original_shape = original_data.shape
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.n_component = n_component
        self.threshold = threshold
        self.norm_func = norm_func
        self.use_hilbert_amplitude = use_hilbert_amplitude
        self.train_fraction = train_fraction
        self.train_mask = train_mask
        self.verbose = verbose

        if self.use_hilbert_amplitude:
            if len(self.original_shape) > 2:
                raise ValueError("Hilbert transform is only supported for 1D or 2D channel images.")
            if self.verbose:
                print("[Hilbert] Applying Hilbert transform and calculating amplitude for each component.")
            # Apply Hilbert transform along the last axis of each component and get amplitude
            self.decomposition = np.array([np.abs(hilbert(comp)) for comp in self.decomposition])

        self.embed_map, self.umap_model = umap_embedding(
            self.decomposition,
            original_data=original_data,
            umap_n_neighbors=umap_n_neighbors,
            umap_min_dist=umap_min_dist,
            n_component=n_component,
            threshold=threshold,
            norm_func=norm_func,
            train_fraction=train_fraction,
            train_mask=train_mask,
            verbose=verbose
        )

    def save_umap_model(self, filename):
        """Save the trained UMAP model to a file.

        Args:
            filename (str): Path to the file where the UMAP model will be saved.
        """
        if self.umap_model is None:
            raise ValueError("UMAP model is not trained")
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.umap_model, f)
            if self.verbose:
                print(f"[UMAP] Model saved to {filename}")
        except IOError as e:
            raise IOError(f"Error saving UMAP model to {filename}: {str(e)}")

    def load_umap_model(self, filename):
        """Load a UMAP model from a file.

        Args:
            filename (str): Path to the file containing the saved UMAP model.
        """
        try:
            with open(filename, 'rb') as f:
                loaded_model = pickle.load(f)
        except IOError as e:
            raise IOError(f"Error loading UMAP model from {filename}: {str(e)}")

        if not isinstance(loaded_model, umap.UMAP):
            raise ValueError(f"Loaded object from {filename} is not a valid UMAP model")

        self.umap_model = loaded_model
        self.n_component = loaded_model.n_components
        if self.verbose:
            print(f"[UMAP] Model loaded from {filename}")

    def compute_new_embeddings(self, new_decomposition=None, new_original_data=None):
        """Compute UMAP embeddings for new decomposition or data using the trained UMAP model.

        Args:
            new_decomposition (numpy.ndarray, optional): New pre-computed decomposition.
            new_original_data (numpy.ndarray, optional): New original data for thresholding and reshaping, or for decomposition if new_decomposition is None.

        Returns:
            list: List of arrays, each a UMAP component reshaped to the original data shape.
        """
        if self.umap_model is None:
            raise ValueError("UMAP model is not trained")
        if (new_decomposition is None and new_original_data is None) or (new_decomposition is not None and new_original_data is None):
            raise ValueError("Either new_decomposition or new_original_data must be provided, and new_original_data is required if new_decomposition is provided")

        if new_decomposition is None:
            if self.decomposition_func is None:
                raise ValueError("decomposition_func must be provided to decompose new_original_data")
            decomposition_result = self.decomposition_func(new_original_data)
            if isinstance(decomposition_result, tuple):
                new_decomposition = np.array(decomposition_result[0])
            else:
                new_decomposition = np.array(decomposition_result)
            original_shape = new_original_data.shape
        else:
            original_shape = new_original_data.shape
            new_decomposition = np.array(new_decomposition)

        if self.verbose:
            print("[Decomposition] Processing completed")

        if self.use_hilbert_amplitude:
            if len(original_shape) > 2:
                raise ValueError("Hilbert transform is only supported for 1D or 2D channel images.")
            if self.verbose:
                print("[Hilbert] Applying Hilbert transform and calculating amplitude for each new component.")
            new_decomposition = np.array([np.abs(hilbert(comp)) for comp in new_decomposition])

        first_dim = new_decomposition.shape[0]
        second_dim = np.prod(original_shape)
        data_input = new_decomposition.reshape(first_dim, second_dim)

        if self.threshold is not None:
            mask = np.ones_like(new_original_data)
            mask[new_original_data < self.threshold] = np.nan
            data_input = data_input * mask.reshape(-1)

        valid_column_indices = np.where(np.all(~np.isnan(data_input), axis=0))[0]
        filtered_data = data_input[:, valid_column_indices]

        umap_data = filtered_data.T
        if self.norm_func is not None:
            umap_data = np.apply_along_axis(self.norm_func, 1, umap_data)

        new_embedding = self.umap_model.transform(umap_data)

        new_embed_map = [new_embedding[:, i].reshape(original_shape) for i in range(new_embedding.shape[1])]

        return new_embed_map


def decompose_and_embed(data, decomposition_method='cdd', decomposition_func=None, umap_n_neighbors=30, umap_min_dist=0.0, n_component=2,
                        threshold=None, cdd_max_n=None, emd_max_imf=-1, msm_filter_sizes='auto', norm_func=None,
                        use_hilbert_amplitude=False, train_fraction=None, train_mask=None, verbose=True):
    """Perform decomposition (CDD, EMD, MSM, AMD, or custom) followed by UMAP dimensionality reduction.

    Args:
        data (numpy.ndarray): Input data array (1D for EMD, multidimensional for CDD/MSM/AMD/custom).
        decomposition_method (str, optional): Decomposition method ('cdd', 'emd', 'msm', 'amd'). Ignored if decomposition_func is provided. Defaults to 'cdd'.
        decomposition_func (callable, optional): Custom function to compute decomposition. Must return a numpy.ndarray or a tuple where the first element is the decomposition. Defaults to None.
        umap_n_neighbors (int, optional): Number of neighbors for UMAP. Defaults to 30.
        umap_min_dist (float, optional): Minimum distance in UMAP embedding. Defaults to 0.0.
        n_component (int, optional): Number of UMAP components. Defaults to 2.
        threshold (float, optional): Threshold to mask data below this value with NaN. Defaults to None.
        cdd_max_n (int, optional): Maximum number of components for CDD or AMD. Defaults to None.
        emd_max_imf (int, optional): Maximum number of IMFs for EMD. Defaults to -1 (all IMFs).
        msm_filter_sizes (tuple or str, optional): Tuple of (base_window, max_scale, spacing) or 'auto' for MSM. Defaults to 'auto'.
        norm_func (callable, optional): Normalization function applied to rows. Defaults to None.
        use_hilbert_amplitude (bool, optional): If True, applies Hilbert transform to get the analytical amplitude of each decomposition component. Defaults to False.
        train_fraction (float, optional): Fraction of data to use for training UMAP (0 to 1). Defaults to None.
        train_mask (numpy.ndarray, optional): Boolean mask to select training data. Defaults to None.
        verbose (bool, optional): Whether to enable verbose output. Defaults to True.

    Returns:
        list: List of UMAP component arrays.
        numpy.ndarray: Decomposition result.
        umap.UMAP: Trained UMAP model.
    """
    if decomposition_func is not None and not callable(decomposition_func):
        raise ValueError("decomposition_func must be a callable function")
    if decomposition_func is None and decomposition_method not in ['cdd', 'emd', 'msm', 'amd']:
        raise ValueError("decomposition_method must be one of 'cdd', 'emd', 'msm', or 'amd' when decomposition_func is None")
    if decomposition_method == 'msm' and decomposition_func is None and msm_filter_sizes is None:
        raise ValueError("msm_filter_sizes must be provided or set to 'auto' for MSM decomposition")
    if decomposition_method == 'msm' and decomposition_func is None and msm_filter_sizes != 'auto' and \
       (not isinstance(msm_filter_sizes, tuple) or len(msm_filter_sizes) != 3):
        raise ValueError("msm_filter_sizes must be 'auto' or a tuple of (base_window, max_scale, spacing)")
    if use_hilbert_amplitude and len(data.shape) > 2:
        raise ValueError("Hilbert transform is only supported for 1D or 2D input data.")

    if decomposition_func is not None:
        decomposition_result = decomposition_func(data)
        if isinstance(decomposition_result, tuple):
            decomposition = np.array(decomposition_result[0])
        else:
            decomposition = np.array(decomposition_result)
    elif decomposition_method == 'cdd':
        decomposition, _ = cdd_decomposition(data, max_n=cdd_max_n)
        decomposition = np.array(decomposition)
    elif decomposition_method == 'emd':
        decomposition = emd_decomposition(data, max_imf=emd_max_imf)
    elif decomposition_method == 'amd':
        decomposition, _ = adaptive_multiscale_decomposition(data, max_n=cdd_max_n)
        decomposition = np.array(decomposition)
    else:  # msm
        if msm_filter_sizes == 'auto':
            msm_filter_sizes = _auto_msm_filter_sizes(data.shape)
            if verbose:
                print(f"[MSM] Automatically selected filter sizes: {msm_filter_sizes}")
        base_window, max_scale, spacing = msm_filter_sizes
        decomposition_list, _ = msm_decomposition(data, base_window, max_scale, spacing)
        decomposition = np.array(decomposition_list)

    if verbose:
        print("[Decomposition] Processing completed")

    decomp_umap = DecompositionUMAP(
        decomposition=decomposition,
        original_data=data,
        decomposition_func=decomposition_func,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        n_component=n_component,
        threshold=threshold,
        norm_func=norm_func,
        use_hilbert_amplitude=use_hilbert_amplitude,
        train_fraction=train_fraction,
        train_mask=train_mask,
        verbose=verbose
    )
    return decomp_umap.embed_map, decomp_umap.decomposition, decomp_umap.umap_model


def decompose_with_existing_model(model_filename, data=None, decomposition=None, decomposition_method='cdd',
                                 decomposition_func=None, threshold=None, cdd_max_n=None, emd_max_imf=-1,
                                 msm_filter_sizes='auto', norm_func=None, use_hilbert_amplitude=False, verbose=True):
    """Apply a pre-trained UMAP model to new data or decomposition, ensuring consistent decomposition.

    Args:
        model_filename (str): Path to the saved UMAP model file.
        data (numpy.ndarray, optional): Input data array (1D for EMD, multidimensional for CDD/MSM/AMD/custom).
        decomposition (numpy.ndarray, optional): Pre-computed decomposition.
        decomposition_method (str, optional): Decomposition method ('cdd', 'emd', 'msm', 'amd'). Ignored if decomposition or decomposition_func is provided. Defaults to 'cdd'.
        decomposition_func (callable, optional): Custom function to compute decomposition. Must return a numpy.ndarray or a tuple where the first element is the decomposition. Defaults to None.
        threshold (float, optional): Threshold to mask data below this value with NaN. Defaults to None.
        cdd_max_n (int, optional): Maximum number of components for CDD or AMD. Defaults to None.
        emd_max_imf (int, optional): Maximum number of IMFs for EMD. Defaults to -1.
        msm_filter_sizes (tuple or str, optional): Tuple of (base_window, max_scale, spacing) or 'auto' for MSM. Defaults to 'auto'.
        norm_func (callable, optional): Normalization function applied to rows. Defaults to None.
        use_hilbert_amplitude (bool, optional): If True, applies Hilbert transform. This must match the setting used to train the model. Defaults to False.
        verbose (bool, optional): Whether to enable verbose output. Defaults to True.

    Returns:
        list: List of UMAP component arrays.
        numpy.ndarray: Decomposition result.
    """
    if (data is None and decomposition is None) or (data is not None and decomposition is not None):
        raise ValueError("Exactly one of data or decomposition must be provided")
    if decomposition_func is not None and not callable(decomposition_func):
        raise ValueError("decomposition_func must be a callable function")
    if decomposition is None and decomposition_func is None and decomposition_method not in ['cdd', 'emd', 'msm', 'amd']:
        raise ValueError("decomposition_method must be one of 'cdd', 'emd', 'msm', or 'amd' when decomposition and decomposition_func are None")
    if decomposition_method == 'msm' and decomposition is None and decomposition_func is None and msm_filter_sizes is None:
        raise ValueError("msm_filter_sizes must be provided or set to 'auto' for MSM decomposition")
    if decomposition_method == 'msm' and decomposition is None and decomposition_func is None and msm_filter_sizes != 'auto' and \
       (not isinstance(msm_filter_sizes, tuple) or len(msm_filter_sizes) != 3):
        raise ValueError("msm_filter_sizes must be 'auto' or a tuple of (base_window, max_scale, spacing)")
    
    if use_hilbert_amplitude:
        if data is not None and len(data.shape) > 2:
            raise ValueError("Hilbert transform is only supported for 1D or 2D input data.")
        if decomposition is not None and len(decomposition[0].shape) > 2:
            raise ValueError("Hilbert transform is only supported for 1D or 2D channel images.")


    # Load UMAP model
    try:
        with open(model_filename, 'rb') as f:
            umap_model = pickle.load(f)
    except IOError as e:
        raise IOError(f"Error loading UMAP model from {model_filename}: {str(e)}")

    if not isinstance(umap_model, umap.UMAP):
        raise ValueError(f"Loaded object from {model_filename} is not a valid UMAP model")

    if verbose:
        print(f"[UMAP] Model loaded from {model_filename}")

    # Determine the shape of the original data for instantiation
    if data is not None:
        original_data_for_init = data
    else: # decomposition is not None
        original_data_for_init = np.zeros(decomposition[0].shape)
    
    dummy_decomposition = np.zeros((1,) + original_data_for_init.shape)

    # Create DecompositionUMAP instance to host the model and methods
    decomp_umap = DecompositionUMAP(
        decomposition=dummy_decomposition,
        original_data=original_data_for_init,
        decomposition_func=decomposition_func,
        threshold=threshold,
        norm_func=norm_func,
        use_hilbert_amplitude=use_hilbert_amplitude,
        verbose=verbose
    )
    decomp_umap.load_umap_model(model_filename)

    # Handle decomposition of new data
    if decomposition is not None:
        final_decomposition = decomposition
        original_data = original_data_for_init
    else:
        if decomp_umap.decomposition_func is not None:
            decomposition_result = decomp_umap.decomposition_func(data)
            if isinstance(decomposition_result, tuple):
                final_decomposition = np.array(decomposition_result[0])
            else:
                final_decomposition = np.array(decomposition_result)
        elif decomposition_method == 'cdd':
            final_decomposition, _ = cdd_decomposition(data, max_n=cdd_max_n)
            final_decomposition = np.array(final_decomposition)
        elif decomposition_method == 'emd':
            final_decomposition = emd_decomposition(data, max_imf=emd_max_imf)
        elif decomposition_method == 'amd':
            final_decomposition, _ = adaptive_multiscale_decomposition(data, max_n=cdd_max_n)
            final_decomposition = np.array(final_decomposition)
        else:  # msm
            if msm_filter_sizes == 'auto':
                msm_filter_sizes = _auto_msm_filter_sizes(data.shape)
                if verbose:
                    print(f"[MSM] Automatically selected filter sizes: {msm_filter_sizes}")
            base_window, max_scale, spacing = msm_filter_sizes
            decomposition_list, _ = msm_decomposition(data, base_window, max_scale, spacing)
            final_decomposition = np.array(decomposition_list)
        original_data = data

    if verbose:
        print("[Decomposition] Processing completed")

    # Compute embeddings using the loaded model and consistent processing
    embed_map = decomp_umap.compute_new_embeddings(
        new_decomposition=final_decomposition,
        new_original_data=original_data
    )

    # The decomposition to return should be the one after Hilbert transform if applied
    if use_hilbert_amplitude:
        final_decomposition = np.array([np.abs(hilbert(comp)) for comp in final_decomposition])


    return embed_map, final_decomposition