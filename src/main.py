import os
import numpy as np
from .utils import load_and_preprocess_graph, plot_network_properties
from .part_a import analyze_basic_properties
from .part_b import compute_degree_distribution
from .part_c import compute_shortest_paths
from .part_d import compute_clustering_coefficients
from .part_e import compute_eigenspectrum
from .part_f import compute_degree_correlations
from .part_g import compute_degree_clustering_relation
from .barabasi_albert import generate_barabasi_albert_graph

def analyze_graph(adj_matrix, graph_name):
    """
    Analyze all network properties and create plots
    """
    properties = {}
    
    # Basic properties (part a)
    properties.update(analyze_basic_properties(adj_matrix))
    
    # Degree distribution (part b)
    properties['degree_dist'] = compute_degree_distribution(adj_matrix)
    
    # Shortest paths (part c)
    shortest_paths = compute_shortest_paths(adj_matrix)
    properties['shortest_paths'] = shortest_paths['paths']
    properties['average_path_length'] = shortest_paths['average_path_length']
    
    # Clustering coefficients (part d)
    clustering = compute_clustering_coefficients(adj_matrix)
    properties['clustering_coeffs'] = clustering['coefficients']
    properties['average_clustering'] = clustering['average_coefficient']
    
    # Eigenspectrum (part e)
    properties['eigenvalues'], properties['spectral_gap'] = compute_eigenspectrum(adj_matrix)
    
    # Degree correlations (part f)
    src_deg, tgt_deg, corr = compute_degree_correlations(adj_matrix)
    properties['degree_corr'] = (src_deg, tgt_deg, corr)
    
    # Degree-clustering relation (part g)
    properties['degree_clustering'] = compute_degree_clustering_relation(adj_matrix)
    
    # Plot everything
    plot_network_properties(graph_name, properties)
    
    return properties

def main():
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Analyze real networks
    datasets = [
        'data/protein.edgelist.txt',  # Protein network
        'data/email.edgelist.txt',  # Email network
        'data/phonecalls.edgelist.txt'   # Phone calls network
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
        print(f"Average clustering: {properties['average_clustering']:.2f}")
        print(f"Average path length: {properties['average_path_length']:.2f}")
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
        print(f"Average clustering: {ba_properties['average_clustering']:.2f}")
        print(f"Average path length: {ba_properties['average_path_length']:.2f}")
        print(f"Spectral gap: {ba_properties['spectral_gap']:.2f}")
        print(f"Degree correlation: {ba_properties['degree_corr'][2]:.2f}")

if __name__ == "__main__":
    main() 