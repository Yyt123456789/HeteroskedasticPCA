import numpy as np

def generate_stationary_ar_coefficients(r, random_seed=1):
    """
    Generate a stationary AR coefficient matrix for a multivariate AR(1) process.
    
    Parameters:
    r (int): The dimension of the AR process.
    
    Returns:
    np.ndarray: An (r, r) AR coefficient matrix with eigenvalues less than 1 in absolute value.
    """
    np.random.seed(random_seed)
    # Randomly generate a matrix
    A = np.random.randn(r, r)
    
    # Perform SVD decomposition
    U, S, Vt = np.linalg.svd(A)
    
    # Scale singular values to be less than 1 to ensure stationarity
    S = np.diag(np.random.uniform(0, 0.3, size=r))
    
    # Reconstruct the matrix with scaled singular values
    A_stationary = U @ S @ Vt
    
    return A_stationary

def generate_multivariate_ar_process(r, n, ar_coefficients=None, noise_variance=1, burn_in=100, random_seed=1):
    np.random.seed(random_seed)
    # Initialize AR coefficients if not provided, ensuring stationarity
    if ar_coefficients is None:
        ar_coefficients = generate_stationary_ar_coefficients(r, random_seed)
    # Check if the coefficient matrix is of the right shape
    if ar_coefficients.shape != (r, r):
        raise ValueError(f"ar_coefficients should have shape ({r}, {r})")

    # Initialize the time series
    ar_process = np.zeros((n + burn_in, r))

    # Generate the noise
    noise = np.random.randn(n + burn_in, r) * np.sqrt(noise_variance)

    # Simulate the AR process
    for t in range(1, n + burn_in):
        ar_process[t] = ar_process[t-1] @ ar_coefficients + noise[t]

    # Discard burn-in period to remove dependence on initial conditions
    ar_process = ar_process[burn_in:]

    return ar_process
