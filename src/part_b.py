import numpy as np

def compute_degree_distribution(adj_matrix):
    """
    Compute degree distribution and fit power law (part b)
    Returns:
    - degrees: unique degree values
    - counts: frequency of each degree
    - slope: power-law exponent
    - intercept: power-law intercept
    """
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    unique_degrees, degree_counts = np.unique(degrees, return_counts=True)
    
    # Remove zero degrees for log-log fit
    mask = unique_degrees > 0
    unique_degrees = unique_degrees[mask]
    degree_counts = degree_counts[mask]
    
    # Fit power law (log-log)
    log_degrees = np.log(unique_degrees)
    log_counts = np.log(degree_counts)
    slope, intercept = np.polyfit(log_degrees, log_counts, 1)
    
    return {
        'degrees': unique_degrees,
        'counts': degree_counts,
        'slope': slope,
        'intercept': intercept
    } 