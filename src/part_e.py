import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse

def compute_eigenspectrum(adj_matrix, k=100):
    """
    Compute eigenvalue spectrum and spectral gap (part e).
    Returns:
    - eigenvalues: array of the first k eigenvalues
    - spectral_gap: difference between the largest and second-largest eigenvalues
    """
    # Convert adjacency matrix to sparse format
    adj_matrix = scipy.sparse.csc_matrix(adj_matrix)
    n = adj_matrix.shape[0]

    if k >= n - 1:
        # For small graphs, use dense eigenvalue computation
        eigenvalues = scipy.sparse.linalg.eigs(adj_matrix.toarray(), k=k, which='LM', return_eigenvectors=False)
        eigenvalues = np.sort(np.real(eigenvalues))[::-1]
    else:
        # For large graphs, compute the top k eigenvalues using sparse computation
        eigenvalues = scipy.sparse.linalg.eigs(adj_matrix, k=k, which='LM', return_eigenvectors=False)
        eigenvalues = np.sort(np.real(eigenvalues))[::-1]

    # Compute spectral gap (difference between largest and second-largest eigenvalues)
    spectral_gap = eigenvalues[0] - eigenvalues[1]

    # Plot the first 100 eigenvalues (log scale to capture wide ranges, if needed)
    plt.figure(figsize=(8, 6))
    plt.plot(eigenvalues[:100], marker='o', linestyle='-', label="Eigenvalues")
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.title("Top 100 Eigenvalues of the Adjacency Matrix")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show()

    return {
        'eigenvalues': eigenvalues[:100],
        'spectral_gap': spectral_gap
    }
