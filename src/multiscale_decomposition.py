import numpy as np
from math import log
from scipy import ndimage
from scipy.ndimage import laplace, gaussian_filter, median_filter

# For CDD (recommended)
try:
    import constrained_diffusion as cdd_external
    CDD_AVAILABLE = True
except ImportError:
    CDD_AVAILABLE = False
    
    
def compute_gradient_k(data, dx=1):
    """
    Compute gradient_k = |grad(data)| / data for an N-dimensional array.
    
    Parameters:
    - data (np.ndarray): N-dimensional array of input data.
    - dx (float or tuple): Spacing between data points for each axis (default=1).
                          If a single float, applies to all axes. If a tuple, must match the number of dimensions.
    - eps (float): Small constant to avoid division by zero (default=1e-10).
    
    Returns:
    - np.ndarray: Gradient magnitude divided by data, same shape as input.
    
    Raises:
    - ValueError: If data is not a NumPy array or if dx length does not match data dimensions.
    """
    # Ensure data is a NumPy array
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a NumPy array")
    
    # Compute gradients along all axes
    gradients = np.gradient(data, dx)
    if len(data.shape) == 1:
        gradients = [gradients]
    
    # Compute gradient magnitude: sqrt(sum(grad_i^2))
    grad_magnitude = np.sqrt(sum(grad**2 for grad in gradients))
    
    # Avoid division by zero by adding a small constant
    return grad_magnitude / (np.abs(data))

def compute_laplacian_k(data, dx=1,mode='reflect'):
    """Compute laplacian_k = (laplace(data) / data)**0.5 on a coarse scale."""
    laplacian = laplace(data, mode=mode) / (dx**2)  # Scale Laplacian by 1/dx^2
    return np.sqrt(np.abs(laplacian / data))


def compute_gradient_magnitude(image):
    """
    Compute the magnitude of the gradient for an n-dimensional image.

    Parameters:
    image : ndarray
        Input n-dimensional image

    Returns:
    gradient_magnitude : ndarray
        Magnitude of the gradient
    """
    # Compute gradients along each axis
    gradients = np.array(ndimage.sobel(image, axis=-1))

    # Calculate magnitude: sqrt(sum(grad_i^2))
    gradient_magnitude = np.sqrt(np.sum(gradients**2, axis=0))

    return gradient_magnitude


def compute_laplacian(image):
    """
    Compute the Laplacian of an n-dimensional image.

    Parameters:
    image : ndarray
        Input n-dimensional image

    Returns:
    laplacian : ndarray
        Laplacian of the image
    """
    laplacian = np.array(ndimage.laplace(image))
    return laplacian


def adaptive_multiscale_decomposition(data, e_rel=3e-2, max_n=None, sm_mode='reflect'):
    """
    Perform constrained diffusion decomposition on n-dimensional data.

    Args:
        data: n-dimensional array
        e_rel: Relative error (smaller e_rel increases accuracy but computational cost). Default: 3e-2
        max_n: Maximum number of channels (if None, calculated automatically). Default: None
        sm_mode: Mode for array boundary extension in convolution ('reflect', 'constant',
                'nearest', 'mirror', 'wrap'). Default: 'reflect'

    Returns:
        tuple: (results, residual)
            - results: n+1 dimensional array where results[i] contains structures of
                      sizes between 2**i and 2**(i+1) pixels
            - residual: Structures too large to be contained in results
    """
    if data.size == 0:
        raise ValueError("Input data array is empty")

    ntot = int(log(min(data.shape)) / log(2) - 1)
    if max_n is not None:
        ntot = min(ntot, max_n)
    print("total number of scales", ntot)

    result = []
    diff_image = data.copy() * 0

    for i in range(ntot):
        channel_image = data.copy() * 0
        scale_end = float(pow(2, i + 1))
        scale_beginning = float(pow(2, i))
        t_end = scale_end**2 / 2
        t_beginning = scale_beginning**2 / 2

        if i == 0:
            delta_t_max = t_beginning * 0.1
        else:
            delta_t_max = t_beginning * e_rel

        niter = int((t_end - t_beginning) / delta_t_max + 0.5)
        delta_t = (t_end - t_beginning) / niter
        kernel_size = np.sqrt(2 * delta_t)

        print("current channel", i, "current scale", 2**i)

        for _ in range(niter):
            smooth_image = gaussian_filter(data, kernel_size, mode=sm_mode)

            gradient_k = compute_gradient_k(data, 1)
            laplacian_k = compute_laplacian_k(data, 1)

            
            laplacian_ratio = laplacian_k / gradient_k
            # input("Press Enter to continue...")
            valid_points = laplacian_ratio > 1
            # print("valid points", np.sum(valid_points), "out of", data.size)

            DEBUG = False
            if DEBUG:
                import matplotlib.pyplot as plt
                plt.figure()
                # plt.plot(laplacian_k,label='Laplacian k')
                # plt.plot(gradient_k,label='Gradient k')
                plt.plot(laplacian_ratio,label='Laplacian/Gradient ratio')
                plt.legend()
                plt.show()
            diff_image = data.copy() * 0
            diff_image[valid_points] = data[valid_points] - smooth_image[valid_points]

            data = data - diff_image
            channel_image += diff_image

        result.append(channel_image)

    residual = data
    return result, residual


def generate_window_sizes(base_window=3, max_scale=3, spacing='power2'):
    """
    Generate window sizes for decomposition based on spacing type.

    Parameters:
    -----------
    base_window : int
        Base window size (must be odd). Default: 3
    max_scale : int
        Number of decomposition levels (excluding approximation). Default: 3
    spacing : str
        Spacing type: 'power2', 'linear', or 'log'. Default: 'power2'

    Returns:
    --------
    window_sizes : list of int
        List of window sizes for each scale
    """
    if spacing == 'power2':
        return [base_window * (2 ** i) for i in range(max_scale)]
    elif spacing == 'linear':
        return [base_window + i * 2 for i in range(max_scale)]
    elif spacing == 'log':
        max_window = base_window * (2 ** max_scale)
        log_space = np.logspace(np.log10(base_window), np.log10(max_window), max_scale, base=10)
        return [(int(np.round(w)) // 2 * 2 + 1) for w in log_space]
    else:
        raise ValueError("Spacing must be 'power2', 'linear', or 'log'")


def multiscale_median_decomposition(arr, base_window=3, max_scale=3, spacing='power2'):
    """
    Perform multiscale median decomposition on an n-dimensional array.

    Parameters:
    -----------
    arr : ndarray
        Input n-dimensional array
    base_window : int
        Base window size for the smallest scale (must be odd). Default: 3
    max_scale : int
        Maximum scale level (number of decomposition levels). Default: 3
    spacing : str
        Spacing type for scales: 'power2', 'linear', or 'log'. Default: 'power2'

    Returns:
    --------
    decomposition : list of ndarray
        List containing detail layers and the coarsest approximation
        Length is max_scale + 1 (details for each scale + approximation)
    window_sizes : list of int
        Window sizes used for each scale
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError("Input must be a NumPy array")
    if not isinstance(base_window, int) or base_window % 2 != 1:
        raise ValueError("Base window size must be an odd integer")
    if not isinstance(max_scale, int) or max_scale < 1:
        raise ValueError("Max scale must be a positive integer")

    window_sizes = generate_window_sizes(base_window, max_scale, spacing)

    decomposition = []
    current_approx = arr.astype(np.float64)

    for window_size in window_sizes:
        next_approx = median_filter(current_approx, size=window_size, mode='nearest')
        detail = current_approx - next_approx
        decomposition.append(detail)
        current_approx = next_approx

    decomposition.append(current_approx)

    return decomposition, window_sizes


def _auto_msm_filter_sizes(data_shape):
    """
    Automatically determine MSM filter sizes based on data shape.

    Args:
        data_shape (tuple): Shape of the input data.

    Returns:
        tuple: (base_window, max_scale, spacing) for MSM decomposition.
    """
    base_window = 3
    min_dim = min(data_shape) if len(data_shape) > 1 else data_shape[0]
    max_scale = max(1, min(5, int(np.log2(min_dim // 2))))
    spacing = 'power2'
    return (base_window, max_scale, spacing)



# ==============================================================================
# --- Public API Functions ---
# These are the main functions intended for users to call.
# ==============================================================================

def cdd_decomposition(data, e_rel=3e-2, max_n=None, sm_mode='reflect'):
    """
    Performs Constrained Diffusion Decomposition (CDD) on n-dimensional data by
    calling the external `constrained-diffusion-decomposition` library.

    Args:
        data (numpy.ndarray): The n-dimensional input array.
        e_rel (float, optional): Relative error, controlling accuracy vs. speed. Defaults to 3e-2.
        max_n (int, optional): Maximum number of decomposition components (scales).
            If None, it is calculated automatically based on the data size.
        sm_mode (str, optional): The boundary mode for convolution. Defaults to 'reflect'.

    Returns:
        tuple: `(results, residual)` where `results` is a list of NumPy arrays
        representing each scale and `residual` is the leftover coarsest scale.

    Raises:
        ImportError: If the `constrained-diffusion-decomposition` library is not installed.
    """
    # Check if the import at the top of the file was successful.
    if not CDD_AVAILABLE:
        # If not, raise a clear error message telling the user what to do.
        raise ImportError(
            "\nThe 'cdd' decomposition method requires the 'constrained-diffusion-decomposition' package.\n"
            "Please install it with the command:\n\n"
            "pip install constrained-diffusion-decomposition\n"
        )
    
    # If the library is available, call it directly and return its result.
    if 'cdd_external' in globals():
        print("--- Using external 'constrained-diffusion-decomposition' library for CDD. ---")
        return cdd_external.constrained_diffusion_decomposition(data, e_rel=e_rel, num_channels=max_n, sm_mode=sm_mode)
    else:
        # This is a fallback, though the ImportError should be caught first.
        raise RuntimeError("CDD library was marked as unavailable after a successful import check.")

def wavelet_decomposition(data, e_rel=3e-2, max_n=None, sm_mode='reflect'):
    """
    Performs Constrained Diffusion Decomposition (CDD) on n-dimensional data by
    calling the external `constrained-diffusion-decomposition` library.

    Args:
        data (numpy.ndarray): The n-dimensional input array.
        e_rel (float, optional): Relative error, controlling accuracy vs. speed. Defaults to 3e-2.
        max_n (int, optional): Maximum number of decomposition components (scales).
            If None, it is calculated automatically based on the data size.
        sm_mode (str, optional): The boundary mode for convolution. Defaults to 'reflect'.

    Returns:
        tuple: `(results, residual)` where `results` is a list of NumPy arrays
        representing each scale and `residual` is the leftover coarsest scale.

    Raises:
        ImportError: If the `constrained-diffusion-decomposition` library is not installed.
    """
    # Check if the import at the top of the file was successful.
    if not CDD_AVAILABLE:
        # If not, raise a clear error message telling the user what to do.
        raise ImportError(
            "\nThe 'cdd' decomposition method requires the 'constrained-diffusion-decomposition' package.\n"
            "Please install it with the command:\n\n"
            "pip install constrained-diffusion-decomposition\n"
        )
    
    # If the library is available, call it directly and return its result.
    if 'cdd_external' in globals():
        print("--- Using external 'constrained-diffusion-decomposition' library for CDD. ---")
        return cdd_external.constrained_diffusion_decomposition(data, e_rel=e_rel, num_channels=max_n, sm_mode=sm_mode,constrained=False)
    else:
        # This is a fallback, though the ImportError should be caught first.
        raise RuntimeError("CDD library was marked as unavailable after a successful import check.")

def emd_decomposition(data, max_imf=-1):
    """
    Perform Empirical Mode Decomposition (EMD) on 1D data.

    Args:
        data: 1D numpy array
        max_imf: Maximum number of Intrinsic Mode Functions (IMFs). Default: -1 (all IMFs)

    Returns:
        ndarray: Array of IMFs
    """
    if np.ndim(data) != 1:
        raise ValueError("EMD requires 1D input data")
    try:
        from PyEMD import EMD
    except ImportError:
        raise ImportError("PyEMD is required for EMD decomposition. Install it with `pip install emd-signal`.")
    emd = EMD()
    return emd.emd(data, max_imf=max_imf)


def msm_decomposition(data, base_window=3, max_scale=5, spacing='power2'):
    """
    Perform multiscale median decomposition on n-dimensional data.

    Args:
        data: n-dimensional array
        base_window: Base window size (must be odd). Default: 3
        max_scale: Maximum scale level. Default: 3
        spacing: Spacing type ('power2', 'linear', 'log'). Default: 'power2'

    Returns:
        tuple: (decomposition, window_sizes)
    """
    return multiscale_median_decomposition(data, base_window, max_scale, spacing)