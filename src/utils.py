import numpy as np
from sklearn.preprocessing import StandardScaler

def max_norm(vector):
    """
    Normalizes a vector by its maximum absolute value.

    If the vector is all zeros, it remains unchanged.

    Args:
        vector (numpy.ndarray): The input vector to be normalized.

    Returns:
        numpy.ndarray: The normalized vector.
    """
    max_val = np.max(np.abs(vector))
    return vector / max_val if max_val > 0 else vector


def standard_scaler_1d(arr_1d, with_mean=True, with_std=True):
    """
    Correctly scale a 1D array: treat it as one feature with many samples.
    """
    arr = np.asarray(arr_1d)
    
    # Handle empty or single-element arrays gracefully
    if arr.size == 0:
        return arr.copy()
    if arr.size == 1:
        return np.array([0.0]) if with_std else arr.copy()
    
    # Reshape to (n_samples, 1) → one feature, many samples
    arr_2d = arr.reshape(-1, 1)
    
    scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
    scaled_2d = scaler.fit_transform(arr_2d)
    
    return scaled_2d.ravel()  # back to 1D