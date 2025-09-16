import numpy as np
from .utils import max_norm

try:
    import umap
    assert umap.UMAP() is not None
except AttributeError:
    import umap.umap_ as umap
    assert umap.UMAP() is not None

def umap_embedding(decomposition, original_data, n_component=2, umap_n_neighbors=30,
                   umap_min_dist=0.0, threshold=None, norm_func=max_norm,
                   train_fraction=None, train_mask=None, verbose=True):
    """
    Performs UMAP dimensionality reduction on decomposed data.

    This function trains a UMAP model on the provided decomposition components. It
    includes options for thresholding data to exclude low-signal regions,
    normalizing feature vectors, and training the model on a smaller subset of
    the data for efficiency.

    Args:
        decomposition (numpy.ndarray): The decomposed data, with shape
            (n_components, ...original_shape).
        original_data (numpy.ndarray): The original, non-decomposed data. Used for
            its shape and for applying the threshold.
        n_component (int, optional): The number of dimensions for the UMAP
            embedding. Defaults to 2.
        umap_n_neighbors (int, optional): The `n_neighbors` parameter for UMAP.
        umap_min_dist (float, optional): The `min_dist` parameter for UMAP.
        threshold (float, optional): A value below which points in `original_data`
            are masked and excluded from the UMAP analysis.
        norm_func (callable, optional): A function to normalize each feature vector.
        train_fraction (float, optional): A fraction of the valid data
            points to use for training the UMAP model.
        train_mask (numpy.ndarray, optional): A boolean array with the same shape as
            `original_data` specifying exactly which points to use for training.
        verbose (bool, optional): If True, prints progress messages.

    Returns:
        tuple[list[np.ndarray], umap.UMAP]:
        - embed_map (list[np.ndarray]): A list of numpy arrays, one for each
          embedding dimension, reshaped to match the `original_data` shape.
        - umap_model (umap.UMAP): The trained UMAP reducer object.
    """
    original_shape = original_data.shape

    if threshold is not None:
        mask = np.ones(original_shape)
        mask[original_data < threshold] = np.nan
        decomposition = decomposition * mask.reshape(1, -1) if mask.ndim == 1 else decomposition * mask

    data_input = decomposition.reshape(decomposition.shape[0], -1)
    valid_column_indices = np.where(np.all(~np.isnan(data_input), axis=0))[0]
    filtered_data = data_input[:, valid_column_indices]
    umap_data = filtered_data.T

    if train_mask is not None:
        if train_mask.shape != original_shape:
            raise ValueError(f"train_mask shape {train_mask.shape} must match {original_shape}")
        train_mask_flat = train_mask.flatten()[valid_column_indices]
        train_indices = np.where(train_mask_flat)[0]
    elif train_fraction is not None:
        if not (0 < train_fraction <= 1):
            raise ValueError("train_fraction must be between 0 and 1")
        n_train = int(umap_data.shape[0] * train_fraction)
        train_indices = np.random.choice(umap_data.shape[0], size=n_train, replace=False)
    else:
        train_indices = np.arange(umap_data.shape[0])

    if verbose:
        print(f"[UMAP] Training on {len(train_indices)} of {umap_data.shape[0]} valid data points.")

    umap_data_train = umap_data[train_indices]

    if norm_func is not None:
        umap_data_train = np.apply_along_axis(norm_func, 1, umap_data_train)

    reducer = umap.UMAP(n_components=n_component, n_neighbors=umap_n_neighbors,
                        min_dist=umap_min_dist, verbose=verbose)
    umap_model = reducer.fit(umap_data_train)

    if train_mask is not None or train_fraction is not None:
        if verbose: print("[UMAP] Transforming the full dataset.")
        umap_data_full = umap_data
        if norm_func is not None:
            umap_data_full = np.apply_along_axis(norm_func, 1, umap_data_full)
        transformed_embedding = umap_model.transform(umap_data_full)
    else:
        transformed_embedding = umap_model.embedding_

    full_embedding = np.full((np.prod(original_shape), n_component), np.nan)
    full_embedding[valid_column_indices] = transformed_embedding
    embed_map = [full_embedding[:, i].reshape(original_shape) for i in range(n_component)]

    return embed_map, umap_model
