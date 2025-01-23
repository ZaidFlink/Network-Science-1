import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

def compute_shortest_paths(adj_matrix, sample_size=1000):
    """
    Compute shortest paths distribution (part c)
    Returns array of path lengths
    """
    n = adj_matrix.shape[0]
    if n > sample_size:
        # Sample nodes for large graphs
        nodes = np.random.choice(n, sample_size, replace=False)
    else:
        nodes = np.arange(n)
        
    paths = []
    for source in nodes:
        dist_matrix = shortest_path(adj_matrix, indices=[source], directed=False)
        finite_paths = dist_matrix[np.isfinite(dist_matrix)]
        paths.extend(finite_paths)
    
    paths = np.array(paths)
    return paths[paths > 0]  # Remove zero-length paths 