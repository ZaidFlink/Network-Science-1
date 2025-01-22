import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse import linalg
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import os

def load_and_preprocess_graph(file_path):
    """
    Load graph data and convert to simple undirected graph
    Returns: scipy sparse matrix
    """
    # Load the data
    edges = np.loadtxt(file_path)
    
    # Create sparse matrix
    n = int(max(edges.max(axis=0)[:2]) + 1)
    adj_matrix = sp.csc_matrix((np.ones(len(edges)), 
                               (edges[:, 0].astype(int), edges[:, 1].astype(int))),
                              shape=(n, n))
    
    # Make symmetric (undirected)
    adj_matrix = adj_matrix + adj_matrix.T
    
    # Remove self-loops and make binary
    adj_matrix.setdiag(0)
    adj_matrix = (adj_matrix != 0).astype(int)
    
    return adj_matrix

def analyze_basic_properties(adj_matrix):
    """
    Compute basic network properties (part a)
    """
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

def compute_degree_distribution(adj_matrix):
    """
    Compute degree distribution and fit power law (part b)
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

def compute_shortest_paths(adj_matrix, sample_size=1000):
    """
    Compute shortest paths distribution (part c)
    """
    n = adj_matrix.shape[0]
    if n > sample_size:
        # Sample nodes for large graphs
        nodes = np.random.choice(n, sample_size, replace=False)
    else:
        nodes = np.arange(n)
        
    paths = []
    for source in nodes:
        dist_matrix = sp.csgraph.shortest_path(adj_matrix, indices=[source], directed=False)
        finite_paths = dist_matrix[np.isfinite(dist_matrix)]
        paths.extend(finite_paths)
    
    paths = np.array(paths)
    return paths[paths > 0]  # Remove zero-length paths

def compute_clustering_coefficients(adj_matrix, sample_size=1000):
    """
    Compute clustering coefficient distribution (part d)
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

def compute_eigenspectrum(adj_matrix, k=100):
    """
    Compute eigenvalue spectrum (part e)
    """
    n = adj_matrix.shape[0]
    if k >= n - 1:
        # For small networks, convert to dense and use standard eigenvalue computation
        eigenvalues = np.linalg.eigvals(adj_matrix.toarray())
        eigenvalues = np.sort(np.real(eigenvalues))[::-1]
    else:
        # For large networks, use sparse eigenvalue computation
        eigenvalues = linalg.eigs(adj_matrix, k=k, return_eigenvectors=False)
        eigenvalues = np.sort(np.real(eigenvalues))[::-1]
    
    spectral_gap = eigenvalues[0] - eigenvalues[1]
    return eigenvalues, spectral_gap

def compute_degree_correlations(adj_matrix):
    """
    Compute degree correlations (part f)
    """
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    rows, cols = adj_matrix.nonzero()
    source_degrees = degrees[rows]
    target_degrees = degrees[cols]
    correlation = np.corrcoef(source_degrees, target_degrees)[0, 1]
    return source_degrees, target_degrees, correlation

def compute_degree_clustering_relation(adj_matrix, sample_size=1000):
    """
    Compute degree vs clustering coefficient relation (part g)
    """
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    clustering_coeffs = compute_clustering_coefficients(adj_matrix, sample_size)
    return degrees[:len(clustering_coeffs)], clustering_coeffs

def generate_barabasi_albert_graph(n, m):
    """
    Generate Barabási-Albert preferential attachment graph
    n: number of nodes
    m: number of edges to attach for each new node
    """
    # Initialize with m+1 nodes (minimum needed for m connections)
    adj_matrix = sp.csc_matrix((n, n), dtype=int)
    
    # Connect initial nodes to form a complete graph
    for i in range(m+1):
        for j in range(i+1, m+1):
            adj_matrix[i, j] = adj_matrix[j, i] = 1
    
    # Add remaining nodes
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    for new_node in range(m+1, n):
        # Calculate probabilities based on degrees
        probs = degrees[:new_node]
        if probs.sum() == 0:
            probs = np.ones_like(probs)
        probs = probs / probs.sum()
        
        # Select m targets
        targets = np.random.choice(new_node, size=m, replace=False, p=probs)
        
        # Add edges
        for target in targets:
            adj_matrix[new_node, target] = adj_matrix[target, new_node] = 1
        
        # Update degrees
        degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    
    return adj_matrix

def plot_network_properties(graph_name, properties):
    """
    Plot all network properties
    """
    plt.figure(figsize=(20, 15))
    
    # Degree distribution
    plt.subplot(231)
    plt.loglog(properties['degree_dist']['degrees'], 
              properties['degree_dist']['counts'], 'bo-', label='Data')
    x_fit = properties['degree_dist']['degrees']
    y_fit = np.exp(properties['degree_dist']['intercept'] + 
                   properties['degree_dist']['slope'] * np.log(x_fit))
    plt.loglog(x_fit, y_fit, 'r--', 
              label=f'Fit (slope={properties["degree_dist"]["slope"]:.2f})')
    plt.xlabel('Degree')
    plt.ylabel('Count')
    plt.title('Degree Distribution')
    plt.legend()
    
    # Shortest paths
    plt.subplot(232)
    plt.hist(properties['shortest_paths'], bins=50, density=True)
    plt.xlabel('Path Length')
    plt.ylabel('Frequency')
    plt.title(f'Shortest Paths\nAvg: {np.mean(properties["shortest_paths"]):.2f}')
    
    # Clustering coefficients
    plt.subplot(233)
    plt.hist(properties['clustering_coeffs'], bins=50, density=True)
    plt.xlabel('Clustering Coefficient')
    plt.ylabel('Frequency')
    plt.title(f'Clustering Coefficients\nAvg: {np.mean(properties["clustering_coeffs"]):.2f}')
    
    # Eigenspectrum
    plt.subplot(234)
    plt.plot(properties['eigenvalues'], 'bo-')
    plt.xlabel('Rank')
    plt.ylabel('Eigenvalue')
    plt.title(f'Eigenvalue Spectrum\nGap: {properties["spectral_gap"]:.2f}')
    
    # Degree correlations
    plt.subplot(235)
    plt.scatter(properties['degree_corr'][0], properties['degree_corr'][1], 
               alpha=0.1, s=1)
    plt.xlabel('Source Degree')
    plt.ylabel('Target Degree')
    plt.title(f'Degree Correlations\nCorr: {properties["degree_corr"][2]:.2f}')
    
    # Degree-clustering relation
    plt.subplot(236)
    plt.scatter(properties['degree_clustering'][0], 
               properties['degree_clustering'][1], alpha=0.1, s=1)
    plt.xlabel('Degree')
    plt.ylabel('Clustering Coefficient')
    plt.title('Degree-Clustering Relation')
    
    plt.tight_layout()
    plt.savefig(f'{graph_name}_properties.png')
    plt.close()

def analyze_graph(adj_matrix, graph_name):
    """
    Analyze all network properties and create plots
    """
    properties = {}
    
    # Basic properties
    properties.update(analyze_basic_properties(adj_matrix))
    
    # Degree distribution
    properties['degree_dist'] = compute_degree_distribution(adj_matrix)
    
    # Shortest paths
    properties['shortest_paths'] = compute_shortest_paths(adj_matrix)
    
    # Clustering coefficients
    properties['clustering_coeffs'] = compute_clustering_coefficients(adj_matrix)
    
    # Eigenspectrum
    properties['eigenvalues'], properties['spectral_gap'] = compute_eigenspectrum(adj_matrix)
    
    # Degree correlations
    src_deg, tgt_deg, corr = compute_degree_correlations(adj_matrix)
    properties['degree_corr'] = (src_deg, tgt_deg, corr)
    
    # Degree-clustering relation
    properties['degree_clustering'] = compute_degree_clustering_relation(adj_matrix)
    
    # Plot everything
    plot_network_properties(graph_name, properties)
    
    return properties

def main():
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Analyze real networks
    datasets = [
        'data/network1.txt',
        'data/network2.txt',
        'data/network3.txt'
    ]
    
    real_properties = {}
    for dataset in datasets:
        name = os.path.splitext(os.path.basename(dataset))[0]
        print(f"\nAnalyzing {name}...")
        
        adj_matrix = load_and_preprocess_graph(dataset)
        properties = analyze_graph(adj_matrix, f"results/{name}")
        real_properties[name] = properties
        
        print(f"Network properties for {name}:")
        print(f"Nodes: {properties['nodes']}")
        print(f"Edges: {properties['edges']}")
        print(f"Components: {properties['n_components']}")
        print(f"Giant component size: {properties['giant_size']}")
        print(f"Power law exponent: {properties['degree_dist']['slope']:.2f}")
        print(f"Average clustering: {np.mean(properties['clustering_coeffs']):.2f}")
        print(f"Spectral gap: {properties['spectral_gap']:.2f}")
        print(f"Degree correlation: {properties['degree_corr'][2]:.2f}")
    
    # Generate and analyze BA networks
    for name, props in real_properties.items():
        n = props['nodes']
        m = max(1, props['edges'] // n)  # Estimate m parameter
        
        print(f"\nGenerating BA network matching {name}...")
        ba_matrix = generate_barabasi_albert_graph(n, m)
        ba_properties = analyze_graph(ba_matrix, f"results/BA_{name}")
        
        print(f"BA network properties:")
        print(f"Nodes: {ba_properties['nodes']}")
        print(f"Edges: {ba_properties['edges']}")
        print(f"Components: {ba_properties['n_components']}")
        print(f"Giant component size: {ba_properties['giant_size']}")
        print(f"Power law exponent: {ba_properties['degree_dist']['slope']:.2f}")
        print(f"Average clustering: {np.mean(ba_properties['clustering_coeffs']):.2f}")
        print(f"Spectral gap: {ba_properties['spectral_gap']:.2f}")
        print(f"Degree correlation: {ba_properties['degree_corr'][2]:.2f}")

if __name__ == "__main__":
    main() 