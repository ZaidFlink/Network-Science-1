import numpy as np
import matplotlib.pyplot as plt

def compute_degree_correlations(adj_matrix, bins=50):
    """
    Compute degree correlations and generate a scatter plot (part f).
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

    # Scatter plot of degree correlations
    plt.figure(figsize=(8, 6))
    plt.hist2d(
        source_degrees,
        target_degrees,
        bins=bins,
        cmap="Blues",
        cmin=1  # To only show bins with data
    )
    plt.colorbar(label="Edge Count (Density)")
    plt.xlabel("Source Degree (d_i)")
    plt.ylabel("Target Degree (d_j)")
    plt.title(f"Degree Correlations (Pearson Correlation: {correlation:.2f})")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xlim(0, source_degrees.max() + 1)
    plt.ylim(0, target_degrees.max() + 1)
    plt.show()

    return {
        'source_degrees': source_degrees,
        'target_degrees': target_degrees,
        'correlation': correlation
    }
