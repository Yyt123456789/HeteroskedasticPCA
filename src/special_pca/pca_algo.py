import numpy as np

def special_pca(Sigma_hat, r, max_iter=100, tol=1e-5):
    """
    Perform the iterative HeteroPCA algorithm without delta and diag helper functions.
    
    Parameters:
    - Sigma_hat: Input covariance matrix.
    - r: Rank of the desired approximation.
    - max_iter: Maximum number of iterations.
    - tol: Convergence tolerance.
    
    Returns:
    - U: Left singular vectors of the final rank-r approximation.
    - N_final: Final matrix after convergence.
    - iter: Number of iterations performed.
    """
    # Initialize: set the diagonal of Sigma_hat to zero by copying Sigma_hat and zeroing its diagonal
    N_t = np.copy(Sigma_hat)
    np.fill_diagonal(N_t, 0)  # Set the diagonal of N_t to zero
    t = 0
    diff = tol + 1
    
    while t < max_iter and diff > tol:
        # Perform SVD on N_t
        U_t, Sigma_t, V_t = np.linalg.svd(N_t, full_matrices=False)
        
        # Best rank-r approximation of N_t
        rank_r_approx = U_t[:, :r] @ np.diag(Sigma_t[:r]) @ V_t[:r, :]
        
        # Update N^(t+1) by setting the diagonal of rank_r_approx and combining it with the off-diagonal part of N_t
        N_t_next = np.copy(rank_r_approx)
        np.fill_diagonal(N_t_next, np.diag(rank_r_approx))  # Set diagonal to that of rank_r_approx
        
        # Convergence criteria: check if the change in the matrix is small
        diff = np.linalg.norm(N_t_next - N_t, ord='fro') / np.linalg.norm(N_t, ord='fro')
        
        # Update N_t and iteration count
        N_t = N_t_next
        t += 1
    
    # Return the rank-r approximation U^(T)
    U_T = U_t[:, :r]
    return U_T
