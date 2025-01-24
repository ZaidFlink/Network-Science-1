import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
import matplotlib.pyplot as plt
import os

data_dir = "./secondaryProject/data/cleaned"
files = ["email.edgelist.txt", "protein.edgelist.txt", "phonecalls.edgelist.txt"]

def analyze_shortest_path(filename, sample_size=1000):

    edge_list = np.loadtxt(os.path.join(data_dir, filename), dtype=int)

    num_nodes = np.max(edge_list) + 1

    adj_matrix = sp.csc_matrix((np.ones(len(edge_list)), (edge_list[:, 0], edge_list[:, 1])), shape=(num_nodes, num_nodes))

    adj_matrix = adj_matrix + adj_matrix.T

    print(f"Computing shortest paths for {filename} (sampling {sample_size} nodes)...")

    sampled_nodes = np.random.choice(num_nodes, sample_size, replace=False)
    distances = shortest_path(csgraph=adj_matrix, indices=sampled_nodes, directed=False)

    distances = distances.flatten()
    distances = distances[np.isfinite(distances)]

    unique_lengths, counts = np.unique(distances, return_counts=True)

    avg_shortest_path_length = np.mean(distances)

    plt.figure(figsize=(10, 6))
    plt.bar(unique_lengths, counts, label="Shortest Path Lengths", color='blue', alpha=0.6)
    plt.xlabel("Shortest Path Length")
    plt.ylabel("Frequency")
    plt.title(f"Shortest Path Length Distribution - {filename}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./secondaryProject/reports/{filename}_shortest_path_length.png")
    plt.show()

    print(f"Graph: {filename}")
    print(f"  - Average shortest path length: {avg_shortest_path_length:.2f}")
    print("-" * 50)

for file in files:
    analyze_shortest_path(file)