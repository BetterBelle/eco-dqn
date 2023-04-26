import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from numba import jit, int64

@jit(nopython=True, parallel=True)
def number_of_uncovered_edges(matrix, spins):
    return matrix * spins + matrix * spins.T

adjacency_matrix = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]])
chosen_nodes = np.array([[1, 0, 0, 0]])
full = number_of_uncovered_edges(adjacency_matrix, chosen_nodes)
print(full)

graph = nx.Graph(adjacency_matrix)
other = nx.Graph(full)
nx.draw_networkx(full)
plt.show()