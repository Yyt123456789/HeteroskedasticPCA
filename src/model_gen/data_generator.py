from .model import *
import numpy as np

def generate_matrix_A(s, p, r, random_seed=1):
    """
    Generate a p x r matrix A where each element is uniformly distributed between -1 and 1.
    The first s columns are scaled by 1/p, and the remaining columns are scaled by 1/sqrt(p).
    
    Parameters:
    s (int): Number of columns to scale by 1/p. Must be less than r.
    p (int): Number of rows in the matrix.
    r (int): Number of columns in the matrix. Must be greater than s.
    
    Returns:
    np.ndarray: A p x r matrix.
    """
    np.random.seed(random_seed)
    # Ensure that s <= r
    if s > r:
        raise ValueError("s must be no greater than r")
    
    # Generate a p x r matrix with values uniformly distributed between -1 and 1
    A = np.random.uniform(low=-1, high=1, size=(p, r))
    
    # Scale the first s columns by 1/p
    A[:, :s] /= p
    
    # Scale the remaining columns by 1/sqrt(p)
    A[:, s:] /= np.sqrt(p)
    
    return A

def apply_matrix_to_ar_process(A, symmetric_matrix=None, ar_process=None, random_seed=1):
    """
    Apply matrix A to the AR process and add noise based on a symmetric matrix (covariance).
    
    Parameters:
    A (np.ndarray): A p x r matrix.
    symmetric_matrix (np.ndarray, optional): A p x p symmetric matrix representing the covariance of the noise.
                                             If None, it defaults to the identity matrix (I).
    ar_process (np.ndarray): The AR process of shape (n, r), where n is the number of samples.
    
    Returns:
    np.ndarray: A (n, p) matrix which is A * ar_process + error, where error is white noise with the given covariance.
    """
    np.random.seed(random_seed)
    # Get dimensions
    p, r = A.shape
    n, ar_r = ar_process.shape
    
    # Ensure dimensions match
    if ar_r != r:
        raise ValueError(f"Matrix A's columns {r} must match AR process dimension {ar_r}")
    
    # If symmetric_matrix is None, use the identity matrix of size p x p
    if symmetric_matrix is None:
        symmetric_matrix = np.eye(p)
    
    # Compute A * ar_process
    result = A @ ar_process.T  # This will be a p x n matrix
    
    # Generate white noise with the given covariance matrix (symmetric_matrix)
    noise = np.random.multivariate_normal(mean=np.zeros(p), cov=symmetric_matrix, size=n).T
    
    # Add noise to the result (transpose to ensure dimensions match n x p)
    final_result = result.T + noise.T  # Result is now n x p
    
    return final_result

def gen_experiment_data( sample_size, strong_f_num, p_dim, total_f_num, random_seed=1):
    A = generate_matrix_A(strong_f_num, p_dim, total_f_num, random_seed= random_seed)
    ar_process = generate_multivariate_ar_process(total_f_num, sample_size, random_seed= random_seed)
    y_process = apply_matrix_to_ar_process(A, ar_process=ar_process, random_seed= random_seed)
    result = {}
    result['A'] = A
    result['ar_process'] = ar_process
    result['y_process'] = y_process
    return result
