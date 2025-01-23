import numpy as np

def compute_clustering_coefficients(adj_matrix, sample_size=1000):
    """
    Compute clustering coefficient distribution (part d)
    Returns array of clustering coefficients
    """
    n = adj_matrix.shape[0]
    if n > sample_size:
        nodes = np.random.choice(n, sample_size, replace=False)
    else:
        nodes = np.arange(n)
        
    coefficients = []
    for node in nodes:
        neighbors = adj_matrix[node].nonzero()[1]
        if len(neighbors) < 2:
            coefficients.append(0)
            continue
            
        possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
        actual_edges = adj_matrix[neighbors][:, neighbors].sum() / 2
        coefficients.append(actual_edges / possible_edges if possible_edges > 0 else 0)
    
    return np.array(coefficients) 