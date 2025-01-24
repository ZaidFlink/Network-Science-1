import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.stats as pearsonr
import os

data_dir = "./secondaryProject/data/cleaned"
files = ["email.edgelist.txt", "protein.edgelist.txt", "phonecalls.edgelist.txt"]

def analyze_degree_correlation(filename, sample_size=5000):
    G = nx.read_edgelist(os.path.join(data_dir, filename), nodetype=int)
    
    # Get degrees of connected nodes (for each edge)
    degrees = dict(G.degree())
    x = []  # degrees of source nodes
    y = []  # degrees of target nodes
    
    for u, v in G.edges():
        x.append(degrees[u])
        y.append(degrees[v])
        # Add reverse edge since graph is undirected
        x.append(degrees[v])
        y.append(degrees[u])
    
    # Calculate correlation
    correlation, _ = pearsonr.pearsonr(x, y)
    
    # Sample for visualization
    if len(x) > sample_size:
        idx = np.random.choice(len(x), sample_size, replace=False)
        x_sample = np.array(x)[idx]
        y_sample = np.array(y)[idx]
    else:
        x_sample = x
        y_sample = y
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x_sample, y_sample, alpha=0.5, label="Degree Correlation", color='blue')
    plt.xlabel("Degree of Node")
    plt.ylabel("Degree of Neighbor")
    plt.title(f"Degree Correlation - {filename}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./secondaryProject/reports/{filename}_degree_correlation.png")
    plt.show()
    
    print(f"Graph: {filename}")
    print(f"  - Degree correlation: {correlation:.2f}")
    print("-" * 50)

for file in files:
    analyze_degree_correlation(file)