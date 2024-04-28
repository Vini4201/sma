

import networkx as nx

import matplotlib.pyplot as plt

G = nx.Graph()

G.add_nodes_from(['a', 'b', 'c', 'd'])

G.nodes()

G.add_edge('a', 'b')
G.add_edge('a', 'c')
G.add_edge('b', 'd')

G.edges()

nx.draw(G)
plt.show()

nx.degree(G)

nx.degree_centrality(G)

nx.shortest_path(G, 'a', 'd')

nx.betweenness_centrality(G)

nx.eigenvector_centrality(G)

"""## Random Graph"""

G = nx.fast_gnp_random_graph(10, 0.5)
nx.draw(G)

nx.draw(G, with_labels=1)

nx.degree(G)

nx.degree_centrality(G)

"""Identifying Most Influential Node"""

m_influential = nx.degree_centrality(G)

for w in sorted(m_influential, key=m_influential.get, reverse=True):
  print(w, m_influential[w])

"""## Activity"""

G = nx.Graph()

G.add_nodes_from(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])

G.nodes()

G.add_edge('A', 'B')
G.add_edge('B', 'C')
G.add_edge('B', 'D')
G.add_edge('C', 'E')
G.add_edge('D', 'E')
G.add_edge('D', 'G')
G.add_edge('E', 'H')
G.add_edge('E', 'G')
G.add_edge('E', 'F')
G.add_edge('G', 'H')

G.edges()

nx.draw(G, with_labels='A')
plt.show()

nx.degree(G)

nx.degree_centrality(G)

nx.betweenness_centrality(G)

nx.eigenvector_centrality(G)

m_influential = nx.degree_centrality(G)

for w in sorted(m_influential, key=m_influential.get, reverse=True):
  print(w, m_influential[w])

n_influential = nx.betweenness_centrality(G)

for w in sorted(n_influential, key=n_influential.get, reverse=True):
  print(w, n_influential[w])

