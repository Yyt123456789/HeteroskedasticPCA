import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
def lag_cov_matrix(X, i):
    X = X.copy()
    n, p = X.shape
    if i >= n:
        raise ValueError("Lag i must be smaller than the number of samples n.")
    if i == 0:
        X_centered = X - np.mean(X, axis=0)
        cov_matrix = (X_centered.T @ X_centered) / n
        return cov_matrix
    
    X_lag = X[i:]     # Lagged version of X
    X_current = X[:-i]  # Non-lagged version (corresponding rows)
    
    X_lag_centered = X_lag - np.mean(X_lag, axis=0)
    X_current_centered = X_current - np.mean(X_current, axis=0)
    
    lag_cov = (X_current_centered.T @ X_lag_centered) / (n - i)
    return lag_cov

def calc_M_matrix(X, start_index = 0, end_index = 1):
    X = X.copy()
    lag_matrices = {}
    
    i = start_index
    lag_matrices[f'{i}_lag_matrix'] = lag_cov_matrix(X,i)
    M = lag_matrices[f'{i}_lag_matrix'] @ lag_matrices[f'{i}_lag_matrix'].T
    while i < end_index:
        i = i + 1
        lag_matrices[f'{i}_lag_matrix'] = lag_cov_matrix(X,i)
        M = M + lag_matrices[f'{i}_lag_matrix'] @ (lag_matrices[f'{i}_lag_matrix'].T)
        
    lag_matrices['M'] = M
    return lag_matrices

def pca_eigenvalue_ratio(matrix, R=None):
    matrix = matrix.copy()
    scaler = StandardScaler()
    matrix = scaler.fit_transform(matrix)
    p, q = matrix.shape
    if p != q:
        raise ValueError("Input matrix must be square.")
    
    # If R is not provided, default to p // 2
    if R is None:
        R = p // 2
    
    # Ensure R is valid
    if R >= p or R <= 1:
        raise ValueError(f"R must be between 1 and {p-1}.")
    
    # Perform PCA (eigen decomposition)
    pca = PCA()
    pca.fit(matrix)
    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_  # Get the eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[sorted_indices].T  # Sort eigenvectors accordingly
    df_eigenvectors = pd.DataFrame(sorted_eigenvectors)

    df_eigenvectors.columns = [f'eigenvector_{i+1}' for i in range(df_eigenvectors.shape[1])]

    # Limit the calculation to the first R eigenvalues
    if len(sorted_eigenvalues) < R:
        raise ValueError(f"R is larger than the number of available eigenvalues ({len(sorted_eigenvalues)}).")
    
    ratios = sorted_eigenvalues[1:R] / sorted_eigenvalues[:R-1]
    min_ratio_index = np.argmin(ratios) + 1  # Adding 1 to get i+1
    
    result = {
        'eigenvalues': sorted_eigenvalues,
        'eigenvectors': df_eigenvectors,
        'min_ratio_index': int(min_ratio_index)
    }
    
    return result