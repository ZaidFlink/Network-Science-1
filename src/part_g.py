import numpy as np
import matplotlib.pyplot as plt

def compute_clustering_coefficients(adj_matrix, sample_size=5000):
    """
    Helper function to compute clustering coefficients for a sample of nodes.
    """
    n = adj_matrix.shape[0]
    if n > sample_size:
        np.random.seed(123)
        nodes = np.random.choice(n, sample_size, replace=False)
    else:
        nodes = np.arange(n)

    coefficients = []
    for node in nodes:
        neighbors = adj_matrix[node].nonzero()[1]
        k = len(neighbors)
        if k < 2:
            coefficients.append(0)
            continue
        possible_edges = k * (k - 1) / 2
        subgraph = adj_matrix[neighbors][:, neighbors]
        actual_edges = subgraph.sum() / 2
        coefficients.append(actual_edges / possible_edges if possible_edges > 0 else 0)

    return np.array(coefficients)

def compute_degree_clustering_relation(adj_matrix, sample_size=1000, bins=50):
    """
    Compute and visualize degree vs clustering coefficient relation (part g).
    Returns:
    - degrees: array of node degrees
    - clustering_coeffs: array of corresponding clustering coefficients
    """
    # Compute degrees for all nodes
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()

    # Compute clustering coefficients for a sampled set of nodes
    clustering_coeffs = compute_clustering_coefficients(adj_matrix, sample_size)

    # Filter degrees to match the sampled nodes
    sampled_degrees = degrees[:len(clustering_coeffs)]

    # Scatter plot with transparency to manage overlapping points
    plt.figure(figsize=(8, 6))
    plt.scatter(sampled_degrees, clustering_coeffs, alpha=0.5, s=10, label="Nodes")
    plt.xlabel("Node Degree (d_i)")
    plt.ylabel("Clustering Coefficient (c_i)")
    plt.title("Degree vs Clustering Coefficient")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.xlim(0, sampled_degrees.max() + 1)
    plt.ylim(0, 1.0)
    plt.show()

    return {
        'degrees': sampled_degrees,
        'clustering_coeffs': clustering_coeffs
    }
