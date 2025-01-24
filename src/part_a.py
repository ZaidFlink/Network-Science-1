import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

def analyze_basic_properties(adj_matrix):
    """
    Compute basic network properties (part a):
    - node/edge sizes
    - number of connected components
    - size of the giant/largest connected component
    """
    # Convert to sparse CSC matrix if not already
    adj_matrix = sp.csc_matrix(adj_matrix)
    
    n_nodes = adj_matrix.shape[0]
    n_edges = int(adj_matrix.sum() / 2)  # Divide by 2 as matrix is symmetric
    
    # Get connected components
    n_components, labels = connected_components(adj_matrix, directed=False)
    
    # Size of giant component
    component_sizes = np.bincount(labels)
    giant_size = component_sizes.max()
    
    return {
        'nodes': n_nodes,
        'edges': n_edges,
        'n_components': n_components,
        'giant_size': giant_size
    }