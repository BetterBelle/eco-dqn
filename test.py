import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import operator as op
import src.envs.score_solver as ss
from src.envs.utils import OptimisationTarget

def predict(network, states, acting_in_reversible_spin_env, allowed_action_state):
    """
    Given a network and environment states, gives actions for each environment. Because the network is on GPU, this allows multiple 
    environments to be evaluated simultaneously.

    TODO: MOVE THIS TO A SOLVER THAT CAN SOLVE MULTIPLE GRAPHS AT ONCE
    """

    qs = network(states)

    if acting_in_reversible_spin_env:
        if qs.dim() == 1:
            actions = [qs.argmax().item()]
        else:
            actions = qs.argmax(1, True).squeeze(1).cpu().numpy()
        return actions
    else:
        if qs.dim() == 1:
            x = (states.squeeze()[:,0] == allowed_action_state).nonzero()
            actions = [x[qs[x].argmax().item()].item()]
        else:
            disallowed_actions_mask = (states[:, :, 0] != allowed_action_state)
            qs_allowed = qs.masked_fill(disallowed_actions_mask, np.finfo(np.float64).min)
            actions = qs_allowed.argmax(1, True).squeeze(1).cpu().numpy()
        return actions

adjacency_matrix = np.array([
    [0, 0, 1, 0, 1, 1],
    [0, 0, 0, 0, 1, 1],
    [1, 0, 0, 1, 1, 0],
    [0, 0, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 1],
    [1, 1, 0, 0, 1, 0]
    ], dtype=np.float64)

spins = np.array([-1., -1., 1., -1., 1., 1.], dtype=np.float64)
vals = - spins * op.matmul(adjacency_matrix * np.array(spins == 1, dtype=np.float64), spins)
val = np.sum(adjacency_matrix * np.array([spins == 1]) * np.array([spins == 1]).T) / 2
solver = ss.ScoreSolverFactory.get(OptimisationTarget.MAX_CLIQUE, False)
solver.set_lower_bound(spins, adjacency_matrix)
solver.set_max_local_reward(spins, adjacency_matrix)
solver.set_quality_normalizer(spins, adjacency_matrix)
solver.set_invalidity_normalizer(spins, adjacency_matrix)
print('Lower Bound')
print(solver._lower_bound)
print('Max Local Reward')
print(solver._max_local_reward)
print('Quality Normalizer')
print(solver._solution_quality_normalizer)
print('Invalidity Normalizer')
print(solver._invalidity_normalizer)
print('Invalidity mask')
print(solver.get_invalidity_degree_mask(spins, adjacency_matrix))
print('Invalidity degree')
print(solver.get_invalidity_degree(spins, adjacency_matrix))
print('Validity mask')
print(solver.get_validity_mask(spins, adjacency_matrix))
print('Score')
print(solver.get_score(spins, adjacency_matrix))
print('Score mask')
print(solver.get_score_mask(spins, adjacency_matrix))
print('Solution')
print(solver.get_solution(spins, adjacency_matrix))
print('Solution Quality')
print(solver.get_solution_quality(spins, adjacency_matrix))
print('Normalized Score')
print(solver.get_normalized_score(spins, adjacency_matrix))
print('Solution Quality Mask')
print(solver.get_solution_quality_mask(spins, adjacency_matrix))
print('Normalized Score Mask')
print(solver.get_normalized_score_mask(spins, adjacency_matrix))
print('Measure')
print(solver._get_measure(spins, adjacency_matrix))

graph = nx.Graph(adjacency_matrix)
nx.draw_networkx(graph)
plt.show()
