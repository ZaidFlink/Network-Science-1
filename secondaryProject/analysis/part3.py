import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigs
from scipy.stats import pearsonr

# Directory setup
synthetic_dir = "./secondaryProject/data/synthetic"
output_dir = "./secondaryProject/reports/"

for directory in [synthetic_dir, output_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Define parameters based on real graphs
graph_params = {
    "email": {"n": 57194, "m": 10},
    "protein": {"n": 2018, "m": 5},
    "phonecalls": {"n": 36595, "m": 8},
}

def generate_barabasi_graph(name, n, m):
    print(f"Generating Barabási-Albert graph for {name} (nodes={n}, edges per new node={m})...")
    G = nx.barabasi_albert_graph(n=n, m=m)
    nx.write_edgelist(G, os.path.join(synthetic_dir, f"{name}_barabasi.edgelist.txt"))
    return G

def compute_basic_properties(G, name):
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    adj_matrix = nx.to_scipy_sparse_array(G)
    num_components, labels = connected_components(csgraph=adj_matrix, directed=False)
    largest_component_size = np.bincount(labels).max()
    
    print(f"{name} - Nodes: {num_nodes}, Edges: {num_edges}")
    print(f"  - Number of connected components: {num_components}")
    print(f"  - Size of largest connected component: {largest_component_size}")

def analyze_degree_distribution(G, name):
    degrees = np.array([d for n, d in G.degree()])
    unique_degrees, counts = np.unique(degrees, return_counts=True)
    
    # Power-law fit (excluding degree 0)
    mask = unique_degrees > 0
    log_degrees = np.log10(unique_degrees[mask])
    log_counts = np.log10(counts[mask])
    slope, intercept = np.polyfit(log_degrees, log_counts, 1)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(log_degrees, log_counts, alpha=0.7, color='blue', label="Degree Distribution")
    plt.plot(log_degrees, slope * log_degrees + intercept, color='red', label=f"Power-law fit (slope={slope:.2f})")
    plt.xlabel('Log(Degree)')
    plt.ylabel('Log(Count)')
    plt.title(f'Degree Distribution - {name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{name}_degree_distribution.png"))
    plt.close()
    
    print(f"  - Power-law slope: {slope:.2f}")

def analyze_shortest_paths(G, name, sample_size=1000):
    print(f"  Computing shortest paths (sampling {sample_size} nodes)...")
    sampled_nodes = np.random.choice(list(G.nodes()), min(sample_size, G.number_of_nodes()), replace=False)
    
    all_distances = []
    for source in sampled_nodes:
        lengths = nx.single_source_shortest_path_length(G, source)
        all_distances.extend(lengths.values())
    
    all_distances = np.array(all_distances)
    avg_shortest_path = np.mean(all_distances)
    
    plt.figure(figsize=(8, 6))
    plt.hist(all_distances, bins=30, color='blue', alpha=0.7)
    plt.xlabel('Shortest Path Length')
    plt.ylabel('Frequency')
    plt.title(f'Shortest Path Distribution - {name}')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{name}_shortest_path_distribution.png"))
    plt.close()
    
    print(f"  - Average shortest path length: {avg_shortest_path:.2f}")

def analyze_clustering_coefficient(G, name):
    print("  Computing clustering coefficients...")
    clustering_coeffs = nx.clustering(G)
    avg_clustering = np.mean(list(clustering_coeffs.values()))
    
    plt.figure(figsize=(8, 6))
    plt.hist(list(clustering_coeffs.values()), bins=30, color='blue', alpha=0.7)
    plt.xlabel('Clustering Coefficient')
    plt.ylabel('Frequency')
    plt.title(f'Clustering Coefficient Distribution - {name}')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{name}_clustering_distribution.png"))
    plt.close()
    
    print(f"  - Average clustering coefficient: {avg_clustering:.4f}")

def analyze_eigenvalues(G, name, num_eigenvalues=100):
    print(f"  Computing {num_eigenvalues} largest eigenvalues...")
    adj_matrix = nx.to_scipy_sparse_array(G)
    eigenvalues, _ = eigs(adj_matrix, k=num_eigenvalues, which='LR')
    eigenvalues = np.real(eigenvalues)
    eigenvalues = np.sort(eigenvalues)[::-1]
    spectral_gap = eigenvalues[0] - eigenvalues[1]
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', color='blue')
    plt.xlabel('Rank')
    plt.ylabel('Eigenvalue')
    plt.title(f'Eigenvalue Spectrum - {name}')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{name}_eigenvalue_spectrum.png"))
    plt.close()
    
    print(f"  - Spectral gap: {spectral_gap:.2f}")

# Main execution
for name, params in graph_params.items():
    print(f"\nAnalyzing {name} network...")
    G = generate_barabasi_graph(name, params["n"], params["m"])
    compute_basic_properties(G, name)
    analyze_degree_distribution(G, name)
    analyze_shortest_paths(G, name)
    analyze_clustering_coefficient(G, name)
    analyze_eigenvalues(G, name)
