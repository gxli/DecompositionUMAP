import numpy as np

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
