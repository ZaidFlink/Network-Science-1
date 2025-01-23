import numpy as np
import scipy.sparse as sp

def generate_barabasi_albert_graph(n, m):
    """
    Generate Barabási-Albert preferential attachment graph
    Parameters:
    - n: number of nodes
    - m: number of edges to attach for each new node
    Returns:
    - adj_matrix: sparse adjacency matrix of the generated graph
    """
    # Initialize with m+1 nodes (minimum needed for m connections)
    adj_matrix = sp.csc_matrix((n, n), dtype=int)
    
    # Connect initial nodes to form a complete graph
    for i in range(m+1):
        for j in range(i+1, m+1):
            adj_matrix[i, j] = adj_matrix[j, i] = 1
    
    # Add remaining nodes
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    for new_node in range(m+1, n):
        # Calculate probabilities based on degrees
        probs = degrees[:new_node]
        if probs.sum() == 0:
            probs = np.ones_like(probs)
        probs = probs / probs.sum()
        
        # Select m targets
        targets = np.random.choice(new_node, size=m, replace=False, p=probs)
        
        # Add edges
        for target in targets:
            adj_matrix[new_node, target] = adj_matrix[target, new_node] = 1
        
        # Update degrees
        degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    
    return adj_matrix 