import numpy as np
from .utils import max_norm

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

def umap_embedding(decomposition, original_data, n_component=2, threshold=None, norm_func=max_norm,
                   train_fraction=None, train_mask=None, verbose=True, umap_params=None):
    """
    Performs UMAP dimensionality reduction on decomposed data.

    This function takes decomposed data, prepares it by handling NaNs and selecting
    a training subset, and then fits a UMAP model using a flexible parameter dictionary.

    Args:
        decomposition (numpy.ndarray): The decomposed data, with shape
            (n_components, ...original_shape).
        original_data (numpy.ndarray): The original data, used for its shape
            and for applying the threshold.
        n_component (int, optional): The number of dimensions for the UMAP
            embedding. This is a required UMAP parameter. Defaults to 2.
        threshold (float, optional): A value below which points in `original_data` are masked.
        norm_func (callable, optional): A function to normalize each feature vector.
        train_fraction (float, optional): Fraction of data for training the UMAP model.
        train_mask (numpy.ndarray, optional): A boolean mask specifying training points.
        verbose (bool, optional): If True, prints progress messages.
        umap_params (dict, optional): A dictionary of keyword arguments to be
            passed directly to the `umap.UMAP` constructor. This allows for
            full control over UMAP's parameters (e.g., `{'n_neighbors': 15,
            'min_dist': 0.0, 'low_memory': True, 'metric': 'cosine'}`). User-provided
            parameters will override the defaults. Defaults to None.

    Returns:
        tuple[list[np.ndarray], umap.UMAP]:
        - embed_map (list[np.ndarray]): List of embedding dimension arrays.
        - umap_model (umap.UMAP): The trained UMAP reducer object.

    Raises:
        ValueError: If the number of training samples is less than `n_neighbors`.
    """
    original_shape = original_data.shape

    if threshold is not None and original_data.any():
        mask = np.ones(original_shape); mask[original_data < threshold] = np.nan
        decomposition = decomposition * mask.reshape(1, -1) if mask.ndim == 1 else decomposition * mask

    data_input = decomposition.reshape(decomposition.shape[0], -1)
    valid_column_indices = np.where(np.all(~np.isnan(data_input), axis=0))[0]
    filtered_data = data_input[:, valid_column_indices]
    umap_data = filtered_data.T

    if train_mask is not None:
        train_mask_flat = train_mask.flatten()[valid_column_indices]
        train_indices = np.where(train_mask_flat)[0]
    elif train_fraction is not None:
        n_train = int(umap_data.shape[0] * train_fraction)
        train_indices = np.random.choice(umap_data.shape[0], size=n_train, replace=False)
    else:
        train_indices = np.arange(umap_data.shape[0])

    if verbose: print(f"[UMAP] Training on {len(train_indices)} of {umap_data.shape[0]} valid data points.")
    umap_data_train = umap_data[train_indices]

    if norm_func is not None:
        umap_data_train = np.apply_along_axis(norm_func, 1, umap_data_train)

    # Prepare the UMAP parameters by merging defaults with user-provided ones
    full_umap_params = {
        'n_components': n_component,
        'n_neighbors': 15,
        'min_dist': 0.1,
        'verbose': verbose
    }
    if umap_params:
        full_umap_params.update(umap_params)

    # Robustness Check: Ensure there are enough samples for the given n_neighbors
    n_samples_for_training = umap_data_train.shape[0]
    if n_samples_for_training < full_umap_params['n_neighbors']:
        raise ValueError(
            f"\n\n[UMAP Error] Cannot train UMAP model.\n"
            f"The number of data points for training ({n_samples_for_training}) is less than `n_neighbors` ({full_umap_params['n_neighbors']}).\n\n"
            "To fix this, consider decreasing `n_neighbors` in the `umap_params` dictionary, using a larger dataset, or reducing the `threshold`."
        )

    # Initialize and fit the UMAP model using the parameter dictionary
    reducer = umap.UMAP(**full_umap_params)
    umap_model = reducer.fit(umap_data_train)

    if train_mask is not None or train_fraction is not None:
        if verbose: print("[UMAP] Transforming the full dataset.")
        transformed_embedding = umap_model.transform(umap_data)
    else:
        transformed_embedding = umap_model.embedding_

    # Reconstruct the embedding map to match the original data shape
    full_embedding = np.full((np.prod(original_shape), n_component), np.nan)
    full_embedding[valid_column_indices] = transformed_embedding
    embed_map = [full_embedding[:, i].reshape(original_shape) for i in range(n_component)]

    return embed_map, umap_model