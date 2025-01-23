import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

def compute_shortest_paths(adj_matrix, sample_size=5000):
    """
    Compute shortest paths distribution (part c)
    Returns:
    - paths: array of shortest path lengths
    - average_path_length: mean of shortest path lengths
    """
    # Convert to sparse matrix
    adj_matrix = sp.csc_matrix(adj_matrix)
    n = adj_matrix.shape[0]

    # Sampling nodes for large graphs
    if n > sample_size:
        np.random.seed(123)  #Reproducibility
        nodes = np.random.choice(n, sample_size, replace=False)
    else:
        nodes = np.arange(n)
    
    paths = []
    
    # Compute shortest paths from sampled nodes
    for source in nodes:
        dist_matrix = shortest_path(adj_matrix, indices=[source], directed=False, unweighted=True)
        finite_paths = dist_matrix[np.isfinite(dist_matrix)]  # Remove infinite distances
        paths.extend(finite_paths[finite_paths > 0])  # Exclude zero-length paths
    
    # Convert paths to numpy array for easier computation
    paths = np.array(paths)
    
    # Calculate average path length
    average_path_length = paths.mean() if len(paths) > 0 else float('inf')
    
    return {
        'paths': paths,
        'average_path_length': average_path_length
    }
