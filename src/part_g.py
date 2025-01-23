import numpy as np
from .part_d import compute_clustering_coefficients

def compute_degree_clustering_relation(adj_matrix, sample_size=1000):
    """
    Compute degree vs clustering coefficient relation (part g)
    Returns:
    - degrees: array of node degrees
    - clustering_coeffs: array of corresponding clustering coefficients
    """
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    clustering_coeffs = compute_clustering_coefficients(adj_matrix, sample_size)
    return degrees[:len(clustering_coeffs)], clustering_coeffs 