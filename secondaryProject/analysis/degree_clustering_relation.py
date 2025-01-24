import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os

data_dir = "./secondaryProject/data/cleaned"
files = ["email.edgelist.txt", "protein.edgelist.txt", "phonecalls.edgelist.txt"]

def analyze_degree_clustering_relation(filename, sample_size=5000):

    G = nx.read_edgelist(os.path.join(data_dir, filename), nodetype=int)

    degrees = dict(G.degree())
    clustering_coeffs = nx.clustering(G)

    degree_values = np.array(list(degrees.values()))
    clustering_coeffs = np.array(list(clustering_coeffs.values()))

    if len(degree_values) > sample_size:
        sampled_indices = np.random.choice(len(degree_values), sample_size, replace=False)
        degree_values = degree_values[sampled_indices]
        clustering_coeffs = clustering_coeffs[sampled_indices]

    plt.figure(figsize=(10, 6))
    plt.scatter(degree_values, clustering_coeffs, alpha=0.5, label="Degree Clustering Relation", color='blue')
    plt.xlabel("Degree")
    plt.ylabel("Clustering Coefficient")
    plt.title(f"Degree Clustering Relation - {filename}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./secondaryProject/reports/{filename}_degree_clustering_relation.png")
    plt.show()

    avg_clustering_coefficient = np.mean(clustering_coeffs)
    print(f"Graph: {filename}")
    print(f"  - Average clustering coefficient: {avg_clustering_coefficient:.2f}")
    print("-" * 50)

for file in files:
    analyze_degree_clustering_relation(file)