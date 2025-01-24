import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import networkx as nx
import os

data_dir = "./secondaryProject/data/cleaned"
files = ["email.edgelist.txt", "protein.edgelist.txt", "phonecalls.edgelist.txt"]

def analyze_graph(filename):

    edge_list = np.loadtxt(os.path.join(data_dir, filename), dtype=int)

    num_nodes = np.max(edge_list) + 1

    adj_matrix = sp.csc_matrix((np.ones(len(edge_list)), (edge_list[:, 0], edge_list[:, 1])),shape=(num_nodes, num_nodes))

    adj_matrix = adj_matrix + adj_matrix.T

    num_edges = adj_matrix.nnz // 2

    num_components, labels = connected_components(csgraph=adj_matrix, directed=False)

    largest_component_size = np.bincount(labels).max()

    print(f"Graph: {filename}")
    print(f"  - Number of nodes: {num_nodes}")
    print(f"  - Number of edges: {num_edges}")          
    print(f"  - Number of connected components: {num_components}")
    print(f"  - Size of the largest connected component: {largest_component_size}")
    print("-" * 50)

for file in files:
    analyze_graph(file)