import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import src.envs.core as ising_env
from src.envs.utils import ( SingleGraphGenerator, MVC_OBSERVABLES, RewardSignal, ExtraAction,
                            OptimisationTarget, SpinBasis )
from experiments.utils import load_graph_set
from src.agents.solver import CplexSolver
from src.networks.mpnn import MPNN
import torch

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

big_graph = np.array([[0 for _ in range(500)] for _ in range(500)])
big_graph[0] = [1 for _ in range(500)]
for i in range(len(big_graph)):
    big_graph[i][0] = 1

big_graph[0][0] = 0

env_args = {'observables':MVC_OBSERVABLES,
                'reward_signal':RewardSignal.BLS,
                'extra_action':ExtraAction.NONE,
                'optimisation_target':OptimisationTarget.MIN_COVER,
                'spin_basis':SpinBasis.SIGNED,
                'norm_rewards':True,
                'memory_length':None,
                'horizon_length':None,
                'stag_punishment':None,
                'basin_reward':True,
                'reversible_spins':True}

test_env = ising_env.make("SpinSystem",
                              SingleGraphGenerator(adjacency_matrix),
                              adjacency_matrix.shape[0]*2,
                              **env_args)

graphs = load_graph_set("_graphs/validation/ER_500spin_p15_100graphs.pkl")
large_test_env = ising_env.make("SpinSystem",
                                SingleGraphGenerator(graphs[0]),
                                graphs[0].shape[0]*2,
                                **env_args)

big_graph = ising_env.make("SpinSystem",
                                SingleGraphGenerator(big_graph),
                                big_graph.shape[0]*2,
                                **env_args)

solver = CplexSolver(big_graph)
solver.reset()
solver.solve()                  


graph = nx.Graph(adjacency_matrix)
# cover = nx.algorithms.approximation.vertex_cover.min_weighted_vertex_cover(graph)
# chosen_cover = np.zeros(4)
# np.put(chosen_cover, list(cover), 1)
# other = nx.Graph(full)
# nx.draw_networkx(full)
nx.draw_networkx(graph)
plt.show()