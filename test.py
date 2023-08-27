import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import operator as op

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

spins = np.array([-1., -1., -1., -1., -1., 1.], dtype=np.float64)
vals = - spins * op.matmul(adjacency_matrix * np.array(spins == 1, dtype=np.float64), spins)
val = np.sum(adjacency_matrix * np.array([spins == 1]) * np.array([spins == 1]).T) / 2

graph = nx.Graph(adjacency_matrix)
nx.draw_networkx(graph)
plt.show()