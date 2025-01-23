import numpy as np
from scipy.sparse import linalg

def compute_eigenspectrum(adj_matrix, k=100):
    """
    Compute eigenvalue spectrum (part e)
    Returns:
    - eigenvalues: array of eigenvalues
    - spectral_gap: difference between largest and second largest eigenvalues
    """
    n = adj_matrix.shape[0]
    if k >= n - 1:
        # For small networks, convert to dense and use standard eigenvalue computation
        eigenvalues = np.linalg.eigvals(adj_matrix.toarray())
        eigenvalues = np.sort(np.real(eigenvalues))[::-1]
    else:
        # For large networks, use sparse eigenvalue computation
        eigenvalues = linalg.eigs(adj_matrix, k=k, return_eigenvectors=False)
        eigenvalues = np.sort(np.real(eigenvalues))[::-1]
    
    spectral_gap = eigenvalues[0] - eigenvalues[1]
    return eigenvalues, spectral_gap 