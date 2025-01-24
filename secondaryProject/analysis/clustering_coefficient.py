import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sp
import os


data_dir = "./secondaryProject/data"
files = ["email.edgelist.txt", "protein.edgelist.txt", "phonecalls.edgelist.txt"]

def analyze_clustering_coefficient(filename, sample_size=1000):
    G = nx.read_edgelist(os.path.join(data_dir, filename), nodetype=int)

    nodes = list(G.nodes())
    sampled_nodes = np.random.choice(nodes, sample_size, replace=False)

    clustering_coeffs = nx.clustering(G, sampled_nodes)

    clustering_values = np.array(list(clustering_coeffs.values()))

    avg_clustering_coefficient = np.mean(clustering_values)

    unique_values, counts = np.unique(clustering_values, return_counts=True)

    plt.figure(figsize=(10, 6))
    plt.bar(unique_values, counts, label="Clustering Coefficients", color='blue', alpha=0.6)
    plt.xlabel("Clustering Coefficient")
    plt.ylabel("Frequency")
    plt.title(f"Clustering Coefficient Distribution - {filename}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./secondaryProject/reports/{filename}_clustering_coefficient.png")
    plt.show()

    print(f"Graph: {filename}")
    print(f"  - Average clustering coefficient: {avg_clustering_coefficient:.2f}")
    print("-" * 50)

for file in files:
    analyze_clustering_coefficient(file)