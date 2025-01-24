import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import os

data_dir = "./secondaryProject/data/cleaned"
files = ["email.edgelist.txt", "protein.edgelist.txt", "phonecalls.edgelist.txt"]

def plot_degree_distribution(filename):
    edge_list = np.loadtxt(os.path.join(data_dir, filename), dtype=int)

    num_nodes = np.max(edge_list) + 1

    adj_matrix = sp.csc_matrix((np.ones(len(edge_list)), (edge_list[:, 0], edge_list[:, 1])), shape=(num_nodes, num_nodes))

    adj_matrix = adj_matrix + adj_matrix.T

    degrees = np.array(adj_matrix.sum(axis=0)).flatten()

    unique_degrees, degree_counts = np.unique(degrees, return_counts=True)

    log_degrees = np.log10(unique_degrees[unique_degrees > 0])
    log_counts = np.log10(degree_counts[unique_degrees > 0])

    slope, intercept = np.polyfit(log_degrees, log_counts, 1)

    plt.figure(figsize=(10, 6))
    plt.scatter(log_degrees, log_counts, label="Degree Distribution", color='blue', alpha=0.6)
    plt.plot(log_degrees, slope * log_degrees + intercept, label='Fit', color='red')
    plt.xlabel('Log Degree')
    plt.ylabel('Log Count')
    plt.title(f'Degree Distribution - {filename}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./secondaryProject/reports/{filename}_degree_distribution.png")
    plt.show()

    print(f"Graph: {filename}")
    print(f"  - Power-law slope: {slope:.2f}")
    print("-" * 50)

for file in files:
    plot_degree_distribution(file)