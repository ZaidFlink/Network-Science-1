import numpy as np

def compute_clustering_coefficients(adj_matrix, sample_size=5000):
    """
    Compute clustering coefficient distribution (part d)
    Returns:
    - coefficients: array of clustering coefficients
    - average_coefficient: mean clustering coefficient
    """
    # Number of nodes in the graph
    n = adj_matrix.shape[0]

    # Sampling nodes for large graphs
    if n > sample_size:
        np.random.seed(123)  #Reproducibility
        nodes = np.random.choice(n, sample_size, replace=False)
    else:
        nodes = np.arange(n)

    coefficients = []

    # Compute clustering coefficient for each sampled node
    for node in nodes:
        neighbors = adj_matrix[node].nonzero()[1]  # Indices of neighbors
        k = len(neighbors)  # Number of neighbors
        
        if k < 2:  # Clustering coefficient is 0 if less than 2 neighbors
            coefficients.append(0)
            continue

        # Calculate possible and actual edges between neighbors
        possible_edges = k * (k - 1) / 2
        subgraph = adj_matrix[neighbors][:, neighbors]
        actual_edges = subgraph.sum() / 2  # Sum of edges in the neighbor subgraph
        clustering_coefficient = actual_edges / possible_edges if possible_edges > 0 else 0
        coefficients.append(clustering_coefficient)

    coefficients = np.array(coefficients)
    
    # Compute average clustering coefficient
    average_coefficient = coefficients.mean() if len(coefficients) > 0 else 0

    return {
        'coefficients': coefficients,
        'average_coefficient': average_coefficient
    }
