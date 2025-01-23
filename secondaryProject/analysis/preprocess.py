import networkx as nx
import os

data_dir = "./secondaryProject/data"
output_dir = "./secondaryProject/data/cleaned"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def preprocess_graph(filename):
    G = nx.read_edgelist(os.path.join(data_dir, filename), nodetype=int)

    G.remove_edges_from(nx.selfloop_edges(G))

    G = nx.Graph(G)

    nx.write_edgelist(G, os.path.join(output_dir, filename))
    print(f"Preprocessed graph saved to {os.path.join(output_dir, filename)}")

files = ["email.edgelist.txt", "protein.edgelist.txt", "phonecalls.edgelist.txt"]
for file in files:
    preprocess_graph(file)