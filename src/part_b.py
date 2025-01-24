import numpy as np
import scipy.sparse as sp

def compute_degree_distribution(adj_matrix):
    """
    Compute degree distribution and fit power law.
    Returns:
    - degrees: unique degree values
    - counts: frequency of each degree
    - slope: power-law exponent
    - intercept: power-law intercept
    """
    # Convert to sparse CSC matrix
    adj_matrix = sp.csc_matrix(adj_matrix)
    
    # Compute degrees
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()  # Sum of rows gives degrees
    
    # Get unique degrees and their frequencies
    unique_degrees, degree_counts = np.unique(degrees, return_counts=True)
    
    # Remove zero degrees for log-log fit
    mask = unique_degrees > 0
    unique_degrees = unique_degrees[mask]
    degree_counts = degree_counts[mask]
    
    # Fit power law using log-log scale
    log_degrees = np.log(unique_degrees)
    log_counts = np.log(degree_counts)
    slope, intercept = np.polyfit(log_degrees, log_counts, 1)
    
    return {
        'degrees': unique_degrees,
        'counts': degree_counts,
        'slope': slope,
        'intercept': intercept
    }
