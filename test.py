import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import operator as op
from numba import jit, float64, int64

def calculate_mvc_rewards_available(matrix, spins):
    ### Explanation: matrix multiplication of adj and spins (vertices) gives number of edges incident on that node
    ### spins == -1 gives vertices that are not in the solution as boolean mask
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

    # We want: If an invalid solution is created, the reward for that flip should be the number of edges that become uncovered (negative)
    # If a valid solution is created, it should be the difference in size of the set
    newly_covered_on_flip = spins * op.matmul(matrix * (spins == -1), spins)
    uncovered_edges = np.sum((matrix != 0) * (spins == -1) * np.array([spins == -1], dtype=np.float64).T) / 2
    total_uncovered_on_flip = uncovered_edges - newly_covered_on_flip
    solution_set_size = sum(spins == 1)


    # First get a boolean mask for validity
    # To verify this, check if the number of uncovered edges minus the number of edges covered on flip (it's negative if it loses edges) is 0
    validity_mask = total_uncovered_on_flip == 0

    # Subtracting spins from solution set size gives you the new set size on flip
    # We then take the total number of nodes and subtract from that the new set size to give you the size of the set not in the solution,
    # this ensures that larger solution sets give smaller rewards (i.e. a set size of 3 with 5 nodes vs a set size of 2 with 5 nodes will now
    # give rewards of 2 and 3 (higher reward for smaller set) instead of 3 and 2)
    # Next, we multiply by the validity mask to keep only the valid solutions
    new_set_size_score = validity_mask * (len(spins) - (solution_set_size - spins))
    
    # Next we want the new number of uncovered edges on flip (i.e. the new degree of invalidity) for each flip
    # This is simply the uncovered edges - newly covered on flip
    new_uncovered_edges =  uncovered_edges - newly_covered_on_flip

    # Score is defined as the set score - new_uncovered edges
    score_on_flip = new_set_size_score - new_uncovered_edges

    # Now, these rewards are relative to the current state, so calculate the score for the current state
    current_validity = (uncovered_edges == 0)
    current_score = current_validity * (len(spins) - solution_set_size) - uncovered_edges

    # Therefore immediate rewards are going to be the difference between the current score and the different scores on each flip
    immediate_rewards_available = score_on_flip - current_score

    return immediate_rewards_available


def _calculate_mvc_score_change(new_spins, matrix, action):
    """
    Given array of new_spins and adjacency matrix, find the score change given the spin "action" was changed
    This is just the -1 * immediate_vertex_covers_available(old_spins), where old spins is just flipping the action
    """
    old_spins = new_spins
    old_spins[action] *= -1

    return -1 * calculate_mvc_rewards_available(matrix, old_spins)[action]

adjacency_matrix = np.array([
    [0, 0, 1, 0, 1, 1],
    [0, 0, 0, 0, 1, 1],
    [1, 0, 0, 1, 1, 0],
    [0, 0, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 1],
    [1, 1, 0, 0, 1, 0]
    ], dtype=np.float64)
chosen_nodes = np.array([-1, -1, -1, -1, 1, 1], dtype=np.float64)


full = calculate_mvc_rewards_available(adjacency_matrix, chosen_nodes)
print(_calculate_mvc_score_change(chosen_nodes, adjacency_matrix, 2))
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