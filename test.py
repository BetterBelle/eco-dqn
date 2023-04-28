import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from numba import jit, float64

# @jit(float64[:](float64[:],float64[:,:]), nopython=True)
def number_of_uncovered_edges(matrix, spins):
    ### Explanation: matrix multiplication of adj and spins (vertices) gives number of edges incident on that node
    ### spins == 1 gives vertices that are not in the solution as boolean mask
    ### Multiplying by matrix gets rid of all edges covered by nodes in solution (0s) without double counting, column wise
    ### That result is essentially a graph removing the edges covered by the solution (but only counting once)
    ### This new matrix multiplied by spins therefore gives the number of edges not covered by another vertex in the solution incident on each vertex
    ### Multiplying this by the state of spins (1s not in solution and -1s in solution) gives the change in number of edges covered on flip
    ### Sum of an adjacency matrix divided by two is the number of edges in that matrix
    ### We want the number of edges that aren't covered by our solution (because our list counts only those edges anyway)
    ### We can do this by multiplying by the spins and it's transpose
    ### If we then take that value and subtract from it the change in covered edges for every vertex, we get a value that represents 
    ### the number of vertices not covered by the solution on each vertex flip
    ### Any value that is a 0 means by flipping that vertex you get a valid cover
    ### We want the immediate reward to be the size of the cover on flip, but have it be -1 when invalid
    
    return np.sum(matrix * (spins == 1) * np.array([spins == 1]).T) / 2 - spins * np.matmul(matrix * (spins == 1), spins)

def _get_immeditate_cuts_avaialable_jit(spins, matrix):
    return spins * np.matmul(matrix, spins)

adjacency_matrix = np.array([
    [0, 0, 1, 0, 1, 1],
    [0, 0, 0, 0, 1, 1],
    [1, 0, 0, 1, 1, 0],
    [0, 0, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 1],
    [1, 1, 0, 0, 1, 0]
    ])
chosen_nodes = np.array([1, 1, 1, -1, -1, -1])


full = number_of_uncovered_edges(adjacency_matrix, np.array(chosen_nodes))
print(_get_immeditate_cuts_avaialable_jit(chosen_nodes, adjacency_matrix))
print(adjacency_matrix)
print(full)


graph = nx.Graph(adjacency_matrix)
# cover = nx.algorithms.approximation.vertex_cover.min_weighted_vertex_cover(graph)
# chosen_cover = np.zeros(4)
# np.put(chosen_cover, list(cover), 1)
# other = nx.Graph(full)
# nx.draw_networkx(full)
nx.draw_networkx(graph)
plt.show()