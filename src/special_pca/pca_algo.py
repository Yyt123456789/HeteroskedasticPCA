import numpy as np
from sklearn.decomposition import PCA

def special_pca(M, r, max_iter=1000, tolerance=1e-5):

    N = np.copy(M)
    np.fill_diagonal(N, 0)
    
    previous_eigenvalues = None
    
    for i in range(max_iter):
        # 对矩阵N做PCA，取前r个主成分
        pca = PCA(n_components=r)
        pca.fit(N)
        eigenvalues = pca.explained_variance_
        eigenvectors = pca.components_.T
        
        # 检查是否达到tolerance
        if previous_eigenvalues is not None:
            if np.abs(previous_eigenvalues - eigenvalues).max() < tolerance:
                break
        
        previous_eigenvalues = eigenvalues
        
        # 用PCA的前r个主成分替换对角线
        new_diag = np.sum(np.square(eigenvectors), axis=1)[:len(N)]  # 新的对角线
        np.fill_diagonal(N, new_diag)
    
    return eigenvectors
