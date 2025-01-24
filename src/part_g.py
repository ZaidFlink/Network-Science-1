import numpy as np
from .part_d import compute_clustering_coefficients

def compute_degree_clustering_relation(adj_matrix, sample_size=1000):
    """
    Compute degree vs clustering coefficient relation (part g).
    Returns:
    - degrees: array of node degrees
    - clustering_coeffs: array of corresponding clustering coefficients
    """
    # Compute degrees for all nodes
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()

    # Compute clustering coefficients for a sampled set of nodes
    clustering_coeffs = compute_clustering_coefficients(adj_matrix, sample_size)['coefficients']

    # Filter degrees to match the sampled nodes
    sampled_degrees = degrees[:len(clustering_coeffs)]

    return sampled_degrees, clustering_coeffs
