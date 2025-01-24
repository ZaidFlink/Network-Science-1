import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs
import os

data_dir = "./secondaryProject/data"
files = ["email.edgelist.txt", "protein.edgelist.txt", "phonecalls.edgelist.txt"]

def analyze_eigenvalue_spectrum(filename, num_eigenvalues=100):
    edge_list = np.loadtxt(os.path.join(data_dir, filename), dtype=int)

    num_nodes = np.max(edge_list) + 1

    adj_matrix = sp.csc_matrix((np.ones(len(edge_list)), (edge_list[:, 0], edge_list[:, 1])), shape=(num_nodes, num_nodes))

    adj_matrix = adj_matrix + adj_matrix.T

    print(f"Computing the top {num_eigenvalues} eigenvalues for {filename}...")

    eigenvalues, _ = eigs(adj_matrix, k=num_eigenvalues, which='LR')

    eigenvalues = np.real(eigenvalues)

    eigenvalues = np.sort(eigenvalues)[::-1]

    spectral_gap = eigenvalues[0] - eigenvalues[1]

    plt.figure(figsize=(10, 6))
    plt.plot(eigenvalues, label="Eigenvalues", color='blue', alpha=0.6)
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.title(f"Eigenvalue Spectrum - {filename}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./secondaryProject/reports/{filename}_eigenvalue_spectrum.png")
    plt.show()

    print(f"Graph: {filename}")
    print(f"  - Spectral gap: {spectral_gap:.2f}")
    print("-" * 50)

for file in files:
    analyze_eigenvalue_spectrum(file)