import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import src.envs.core as ising_env
from src.envs.utils import ( SingleGraphGenerator, MVC_OBSERVABLES, RewardSignal, ExtraAction,
                            OptimisationTarget, SpinBasis )
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

network_fn = MPNN
network_args = {
    'n_layers': 3,
    'n_features': 64,
    'n_hid_readout': [],
    'tied_weights': False
}

test_env = ising_env.make("SpinSystem",
                              SingleGraphGenerator(adjacency_matrix),
                              adjacency_matrix.shape[0]*2,
                              **env_args)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.device(device)
print("Set torch default device to {}.".format(device))

network = network_fn(n_obs_in=test_env.observation_space.shape[1],
                        **network_args).to(device)

# network.load_state_dict(torch.load('ER_20spin/eco/min_cover/network/network_best.pth',map_location=device))
for param in network.parameters():
    param.requires_grad = False
network.eval()

print("Sucessfully created agent with pre-trained MPNN.\nMPNN architecture\n\n{}".format(repr(network)))

obs_batch = test_env.reset([-1] * test_env.n_spins)
done = False
while not done:
    obs_batch = torch.FloatTensor(np.array(obs_batch)).to(device)
    action = predict(network, obs_batch, test_env.reversible_spins, test_env.get_allowed_action_states())[0]
    obs, rew, done, info = test_env.step(action)
    obs_batch = obs


graph = nx.Graph(adjacency_matrix)
# cover = nx.algorithms.approximation.vertex_cover.min_weighted_vertex_cover(graph)
# chosen_cover = np.zeros(4)
# np.put(chosen_cover, list(cover), 1)
# other = nx.Graph(full)
# nx.draw_networkx(full)
nx.draw_networkx(graph)
plt.show()