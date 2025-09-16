import numpy as np
from scipy.integrate import odeint


def add_gaussian_blobs(existing_array, num_gaussians=2, min_sigma=10, max_sigma=10, 
                       min_amplitude=0.6, max_amplitude=0.6, margin=0.15):
    """
    Add random Gaussian distributions to an existing array
    
    Parameters:
        existing_array: The input array
        num_gaussians: Number of Gaussian distributions to add
        min_sigma: Minimum standard deviation
        max_sigma: Maximum standard deviation
        min_amplitude: Minimum amplitude
        max_amplitude: Maximum amplitude
        margin: Minimum distance ratio from edges (0-1)
    """
    # Get array dimensions
    size = existing_array.shape[0]
    
    # Create coordinate grid
    x = np.linspace(0, size-1, size)
    y = np.linspace(0, size-1, size)
    x, y = np.meshgrid(x, y)
    
    # Copy the original array to avoid modifying it
    result = existing_array.copy()
    
    for _ in range(num_gaussians):
        # Randomly generate center point (considering margin)
        margin_pixels = int(size * margin)
        mean_x = np.random.uniform(margin_pixels, size - margin_pixels)
        mean_y = np.random.uniform(margin_pixels, size - margin_pixels)
        
        # Randomly generate standard deviation
        sigma_x = np.random.uniform(min_sigma, max_sigma)
        sigma_y = np.random.uniform(min_sigma, max_sigma)
        
        # Randomly generate amplitude
        amplitude = np.random.uniform(min_amplitude, max_amplitude)
        
        # Compute 2D Gaussian distribution
        gaussian = amplitude * np.exp(-(
            ((x - mean_x) ** 2) / (2 * sigma_x ** 2) + 
            ((y - mean_y) ** 2) / (2 * sigma_y ** 2)
        ))
        
        # Add to result
        result += gaussian
    
    return result

def generate_turing_pattern(
    N=256,
    Du=0.2,
    Dv=0.1,
    F=0.04,
    k=0.06,
    iterations=10000,
    dt=1.0,
    seed=1
):
    """
    Generate a 2D Turing pattern using the Gray-Scott reaction-diffusion model.
    
    Parameters:
        N (int): Grid size (N x N)
        Du (float): Diffusion coefficient for U
        Dv (float): Diffusion coefficient for V
        F (float): Feed rate
        k (float): Removal rate
        iterations (int): Number of simulation steps
        dt (float): Time step for integration
        seed (int): Random seed for reproducibility
    
    Returns:
        ndarray: Final concentration of V
    """
    # Validate inputs
    if N <= 0 or iterations <= 0:
        raise ValueError("N and iterations must be positive")
    
    # Initialize arrays
    U = np.ones((N, N), dtype=np.float64)
    V = np.zeros((N, N), dtype=np.float64)
    
    # Set random initial conditions in the center
    np.random.seed(seed)
    center = slice(N//4, 3*N//4)
    V[center, center] = np.random.uniform(0.2, 0.6, (N//2, N//2))
    
    # Define Laplacian function
    def laplacian(Z):
        return (
            np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0) +
            np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1) - 4 * Z
        ) / (dx := 1.0) ** 2
    
    # Simulate reaction-diffusion
    for _ in range(iterations):
        UVV = U * V**2
        U += dt * (Du * laplacian(U) - UVV + F * (1 - U))
        V += dt * (Dv * laplacian(V) + UVV - (F + k) * V)
        U = np.clip(U, 0, 1)
        V = np.clip(V, 0, 1)
    
    return V

def generate_fractal(
    size=256,
    beta=2.0,
    min_val=0.0,
    max_val=1.0,
    seed=None
):
    """
    Generate a 2D array with fractal noise having a power spectrum p(k) ~ k^-beta.
    
    Parameters:
        size (int): Size of the 2D array (must be power of 2 for FFT)
        beta (float): Power spectrum exponent
        min_val (float): Minimum value of output range
        max_val (float): Maximum value of output range
        seed (int, optional): Random seed for reproducibility
    
    Returns:
        ndarray: size x size array with fractal noise scaled to [min_val, max_val]
    """
    # Validate size is power of 2
    if not (size & (size - 1) == 0):
        raise ValueError("Size must be a power of 2")
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Create frequency grid
    freqs = np.fft.fftfreq(size, d=1.0/size)
    kx, ky = np.meshgrid(freqs, freqs)
    k = np.sqrt(kx**2 + ky**2)
    k[0, 0] = 1e-10  # Avoid division by zero
    
    # Generate fractal noise
    amplitude = k ** (-beta / 2.0)
    phase = np.random.uniform(0, 2 * np.pi, (size, size))
    fourier = amplitude * np.exp(1j * phase)
    fourier[0, 0] = 0  # Zero DC component
    noise = np.fft.ifft2(fourier).real
    
    # Normalize to specified range
    noise_min, noise_max = noise.min(), noise.max()
    if noise_max == noise_min:
        return np.full((size, size), min_val, dtype=np.float64)
    noise = min_val + (max_val - min_val) * (noise - noise_min) / (noise_max - noise_min)
    
    return noise.astype(np.float64)

def generate_turing_with_gaussian(
    N=256,
    Du=0.2,
    Dv=0.1,
    F=0.04,
    k=0.06,
    iterations=10000,
    dt=1.0,
    seed=1,
    num_gaussians=2,
    min_sigma=10,
    max_sigma=10,
    min_amplitude=0.5,
    max_amplitude=0.5,
    margin=0.15
):
    """
    Generate a 2D Turing pattern and add random Gaussian blobs.
    
    Parameters:
        N (int): Grid size (N x N)
        Du (float): Diffusion coefficient for U
        Dv (float): Diffusion coefficient for V
        F (float): Feed rate
        k (float): Removal rate
        iterations (int): Number of simulation steps
        dt (float): Time step for integration
        seed (int): Random seed for reproducibility
        num_gaussians (int): Number of Gaussian distributions to add
        min_sigma (float): Minimum standard deviation for Gaussians
        max_sigma (float): Maximum standard deviation for Gaussians
        min_amplitude (float): Minimum amplitude for Gaussians
        max_amplitude (float): Maximum amplitude for Gaussians
        margin (float): Minimum distance ratio from edges for Gaussians (0-1)
    
    Returns:
        ndarray: Turing pattern with added Gaussian blobs
    """
    # Validate inputs
    if N <= 0 or iterations <= 0:
        raise ValueError("N and iterations must be positive")
    if num_gaussians < 0:
        raise ValueError("num_gaussians must be non-negative")
    if min_sigma <= 0 or max_sigma <= 0:
        raise ValueError("min_sigma and max_sigma must be positive")
    if margin < 0 or margin > 0.5:
        raise ValueError("margin must be between 0 and 0.5")
    
    # Generate Turing pattern
    turing_pattern = generate_turing_pattern(
        N=N, Du=Du, Dv=Dv, F=F, k=k, iterations=iterations, dt=dt, seed=seed
    )
    
    # Add Gaussian blobs
    result = add_gaussian_blobs(
        turing_pattern, num_gaussians, min_sigma, max_sigma, min_amplitude, max_amplitude, margin
    )
    
    return result, turing_pattern, result-turing_pattern

def generate_fractal_with_gaussian(
    size=256,
    beta=2.0,
    min_val=0.0,
    max_val=1.0,
    seed=None,
    num_gaussians=2,
    min_sigma=3,
    max_sigma=5,
    min_amplitude=0.7,
    max_amplitude=0.7,
    margin=0.15
):
    """
    Generate a 2D fractal noise array and add random Gaussian blobs.
    
    Parameters:
        size (int): Size of the 2D array (must be power of 2 for FFT)
        beta (float): Power spectrum exponent
        min_val (float): Minimum value of output range
        max_val (float): Maximum value of output range
        seed (int, optional): Random seed for reproducibility
        num_gaussians (int): Number of Gaussian distributions to add
        min_sigma (float): Minimum standard deviation for Gaussians
        max_sigma (float): Maximum standard deviation for Gaussians
        min_amplitude (float): Minimum amplitude for Gaussians
        max_amplitude (float): Maximum amplitude for Gaussians
        margin (float): Minimum distance ratio from edges for Gaussians (0-1)
    
    Returns:
        ndarray: Fractal noise with added Gaussian blobs
    """
    # Validate inputs
    if not (size & (size - 1) == 0):
        raise ValueError("Size must be a power of 2")
    if num_gaussians < 0:
        raise ValueError("num_gaussians must be non-negative")
    if min_sigma <= 0 or max_sigma <= 0:
        raise ValueError("min_sigma and max_sigma must be positive")
    if margin < 0 or margin > 0.5:
        raise ValueError("margin must be between 0 and 0.5")
    
    # Generate fractal noise
    fractal_noise = generate_fractal(size=size, beta=beta, min_val=min_val, max_val=max_val, seed=seed)
    
    # Add Gaussian blobs
    result = add_gaussian_blobs(
        fractal_noise, num_gaussians, min_sigma, max_sigma, min_amplitude, max_amplitude, margin
    )
    
    return result, fractal_noise, result-fractal_noise



def generate_lorenz_system(initial_state=[1.0, 1.0, 1.0], sigma=10.0, rho=28.0, beta=8.0/3.0, t_max=40, num_points=10000):
    """
    Generate the Lorenz system trajectory in 3D XYZ space.
    
    Parameters:
    initial_state : list, initial conditions [x0, y0, z0]
    sigma, rho, beta : float, Lorenz system parameters
    t_max : float, maximum time for simulation
    num_points : int, number of time points
    """
    # Lorenz system differential equations
    def lorenz(state, t):
        x, y, z = state
        dx_dt = sigma * (y - x)
        dy_dt = x * (rho - z) - y
        dz_dt = x * y - beta * z
        return [dx_dt, dy_dt, dz_dt]

    # Time array
    t = np.linspace(0, t_max, num_points)

    # Solve ODE
    solution = odeint(lorenz, initial_state, t)

    # Extract x, y, z
    x, y, z = solution.T
    
    return t, x, y, z
