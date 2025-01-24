import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

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

def plot_network_properties(graph_name, properties):
    """
    Plot all network properties
    """
    plt.figure(figsize=(20, 15))
    
    # Degree distribution (scatter plot)
    plt.subplot(231)
    plt.scatter(properties['degree_dist']['degrees'], 
               properties['degree_dist']['counts'], 
               alpha=0.6, c='blue', label='Data')
    # Add power law fit line
    x_fit = properties['degree_dist']['degrees']
    y_fit = np.exp(properties['degree_dist']['intercept'] + 
                   properties['degree_dist']['slope'] * np.log(x_fit))
    plt.plot(x_fit, y_fit, 'r--', 
            label=f'Fit (slope={properties["degree_dist"]["slope"]:.2f})')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Degree')
    plt.ylabel('Count')
    plt.title('Degree Distribution')
    plt.legend()
    
    # Shortest paths (histogram)
    plt.subplot(232)
    plt.hist(properties['shortest_paths'], bins=50, density=True, alpha=0.7)
    plt.xlabel('Path Length')
    plt.ylabel('Frequency')
    plt.title(f'Shortest Paths\nAvg: {np.mean(properties["shortest_paths"]):.2f}')
    
    # Clustering coefficients (histogram)
    plt.subplot(233)
    plt.hist(properties['clustering_coeffs'], bins=50, density=True, alpha=0.7)
    plt.xlabel('Clustering Coefficient')
    plt.ylabel('Frequency')
    plt.title(f'Clustering Coefficients\nAvg: {np.mean(properties["clustering_coeffs"]):.2f}')
    
    # Eigenspectrum (scatter plot)
    plt.subplot(234)
    plt.scatter(range(len(properties['eigenvalues'])), 
               properties['eigenvalues'], 
               alpha=0.6, c='blue', s=20)
    plt.xlabel('Rank')
    plt.ylabel('Eigenvalue')
    plt.title(f'Eigenvalue Spectrum\nGap: {properties["spectral_gap"]:.2f}')
    
    # Degree correlations (scatter plot)
    plt.subplot(235)
    plt.scatter(properties['degree_corr'][0], 
               properties['degree_corr'][1], 
               alpha=0.1, s=1, c='blue')
    plt.xlabel('Source Degree')
    plt.ylabel('Target Degree')
    plt.title(f'Degree Correlations\nCorr: {properties["degree_corr"][2]:.2f}')
    
    # Degree-clustering relation (scatter plot)
    plt.subplot(236)
    plt.scatter(properties['degree_clustering'][0], 
               properties['degree_clustering'][1], 
               alpha=0.1, s=1, c='blue')
    plt.xlabel('Degree')
    plt.ylabel('Clustering Coefficient')
    plt.title('Degree-Clustering Relation')
    
    plt.tight_layout()
    plt.savefig(f'{graph_name}_properties.png', dpi=300, bbox_inches='tight')
    plt.close() 