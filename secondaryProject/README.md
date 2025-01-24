results from basic_graph_properties.py:

Here we completed the assignment using networkx to compare our results with the results using networkx. 
In this file, we have the outputs using networkx. Although we see minute differences in the results, this is due to the sampling of nodes which is not the same everytime we run, and we used 1000 samples instead of 5000 and we still saw very similar results for all the parts. Indicating to us that the results are very stable and our implementation is correct.

**Results**

**part 1**

(venv) venvsaifalami@saifs-MacBook-Pro Network-Science-1 % python secondaryProject/analysis/basic_graph_properties.py
Graph: email.edgelist.txt
  - Number of nodes: 57194
  - Number of edges: 92442
  - Number of connected components: 190
  - Size of the largest connected component: 56576
--------------------------------------------------
Graph: protein.edgelist.txt
  - Number of nodes: 2018
  - Number of edges: 2705
  - Number of connected components: 185
  - Size of the largest connected component: 1647
--------------------------------------------------
Graph: phonecalls.edgelist.txt
  - Number of nodes: 36595
  - Number of edges: 56853
  - Number of connected components: 2463
  - Size of the largest connected component: 30420

**part 2**

Graph: email.edgelist.txt
  - Power-law slope: -1.44
--------------------------------------------------
Graph: protein.edgelist.txt
  - Power-law slope: -1.80
--------------------------------------------------
Graph: phonecalls.edgelist.txt
  - Power-law slope: -3.15

**part 3**

Graph: email.edgelist.txt
  - Average shortest path length: 3.93
--------------------------------------------------
Computing shortest paths for protein.edgelist.txt (sampling 1000 nodes)...
Graph: protein.edgelist.txt
  - Average shortest path length: 5.55
--------------------------------------------------
Computing shortest paths for phonecalls.edgelist.txt (sampling 1000 nodes)...
Graph: phonecalls.edgelist.txt
  - Average shortest path length: 10.08
--------------------------------------------------


**part 4**


Graph: email.edgelist.txt
  - Average clustering coefficient: 0.03
--------------------------------------------------
Graph: protein.edgelist.txt
  - Average clustering coefficient: 0.05
--------------------------------------------------
Graph: phonecalls.edgelist.txt
  - Average clustering coefficient: 0.13
--------------------------------------------------


**part 5**
Computing the top 100 eigenvalues for protein.edgelist.txt...
Graph: email.edgelist.txt
  - Spectral gap: 38.24
--------------------------------------------------
Computing the top 100 eigenvalues for protein.edgelist.txt...
Graph: protein.edgelist.txt
  - Spectral gap: 3.65
--------------------------------------------------
Computing the top 100 eigenvalues for phonecalls.edgelist.txt...
Graph: phonecalls.edgelist.txt
  - Spectral gap: 4.84
--------------------------------------------------

**part 6**

2025-01-23 20:29:30.564 Python[5044:16013211] +[IMKInputSession subclass]: chose IMKInputSession_Modern
Graph: email.edgelist.txt
  - Degree correlation: -0.08
--------------------------------------------------
Graph: protein.edgelist.txt
  - Degree correlation: -0.08
--------------------------------------------------
Graph: phonecalls.edgelist.txt
  - Degree correlation: 0.21

  **part 7**

  Graph: email.edgelist.txt
  - Average clustering coefficient: 0.04
--------------------------------------------------
Graph: protein.edgelist.txt
  - Average clustering coefficient: 0.05
--------------------------------------------------
Graph: phonecalls.edgelist.txt
  - Average clustering coefficient: 0.14
--------------------------------------------------



**Question 3:**

Analyzing email network...
Generating Barabási-Albert graph for email (nodes=57194, edges per new node=10)...
email - Nodes: 57194, Edges: 571840
  - Number of connected components: 1
  - Size of largest connected component: 57194
  - Power-law slope: -2.11
  Computing shortest paths (sampling 1000 nodes)...
  - Average shortest path length: 3.49
  Computing clustering coefficients...
  - Average clustering coefficient: 0.0029
  Computing 100 largest eigenvalues...
  - Spectral gap: 22.52

Analyzing protein network...
Generating Barabási-Albert graph for protein (nodes=2018, edges per new node=5)...
protein - Nodes: 2018, Edges: 10065
  - Number of connected components: 1
  - Size of largest connected component: 2018
  - Power-law slope: -1.95
  Computing shortest paths (sampling 1000 nodes)...
  - Average shortest path length: 3.19
  Computing clustering coefficients...
  - Average clustering coefficient: 0.0260
  Computing 100 largest eigenvalues...
  - Spectral gap: 9.71

Analyzing phonecalls network...
Generating Barabási-Albert graph for phonecalls (nodes=36595, edges per new node=8)...
phonecalls - Nodes: 36595, Edges: 292696
  - Number of connected components: 1
  - Size of largest connected component: 36595
  - Power-law slope: -2.10
  Computing shortest paths (sampling 1000 nodes)...
  - Average shortest path length: 3.57
  Computing clustering coefficients...
  - Average clustering coefficient: 0.0035
  Computing 100 largest eigenvalues...
  - Spectral gap: 18.62

