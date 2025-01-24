import numpy as np

def compute_degree_correlations(adj_matrix):
    """
    Compute degree correlations (part f).
    Returns:
    - source_degrees: degrees of source nodes for each edge
    - target_degrees: degrees of target nodes for each edge
    - correlation: Pearson correlation coefficient between source and target degrees
    """
    # Compute node degrees
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()

    # Extract edges (nonzero entries)
    rows, cols = adj_matrix.nonzero()
    source_degrees = degrees[rows]
    target_degrees = degrees[cols]

    # Compute Pearson correlation coefficient
    correlation = np.corrcoef(source_degrees, target_degrees)[0, 1]

    return source_degrees, target_degrees, correlation
