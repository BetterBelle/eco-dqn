import os

import matplotlib.pyplot as plt
import torch
import numpy as np

import sys
sys.path.insert(0, '/home/cloudan/Documents/School/Project/eco-dqn')

import src.envs.core as ising_env
from experiments.utils import test_network, load_graph_set
from src.envs.utils import (SingleGraphGenerator,
                            RewardSignal, ExtraAction,
                            OptimisationTarget, SpinBasis,
                            MVC_OBSERVABLES)
from src.networks.mpnn import MPNN
from src.agents.solver import CplexSolver, CoverMatching, NetworkXMinCoverSolver, Greedy
from src.envs.spinsystem import SpinSystemBase

try:
    import seaborn as sns
    plt.style.use('seaborn')
except ImportError:
    pass

def run(save_loc="ER_60spin/eco/min_cover",
        graph_save_loc="_graphs/validation/ER_20spin_p15_100graphs.pkl",
        batched=True,
        max_batch_size=None):

    print("\n----- Running {} -----\n".format(os.path.basename(__file__)))

    graphs_40_loc = "_graphs/validation/ER_40spin_p15_100graphs.pkl"
    graphs_60_loc = "_graphs/validation/ER_60spin_p15_100graphs.pkl"
    graphs_80_loc = "_graphs/validation/ER_80spin_p15_100graphs.pkl"
    graphs_100_loc = "_graphs/validation/ER_100spin_p15_100graphs.pkl"
    graphs_200_loc = "_graphs/validation/ER_200spin_p15_100graphs.pkl"
    graphs_500_loc = "_graphs/validation/ER_500spin_p15_100graphs.pkl"

    ####################################################
    # NETWORK LOCATION
    ####################################################

    data_folder = os.path.join(save_loc, 'data')
    network_folder = os.path.join(save_loc, 'network')

    print("data folder :", data_folder)
    print("network folder :", network_folder)

    network_save_path = os.path.join(network_folder, 'network_best.pth')

    print("network params :", network_save_path)

    ####################################################
    # NETWORK SETUP
    ####################################################

    network_fn = MPNN
    network_args = {
        'n_layers': 3,
        'n_features': 64,
        'n_hid_readout': [],
        'tied_weights': False
    }

    ####################################################
    # SET UP ENVIRONMENTAL AND VARIABLES
    ####################################################

    step_factor = 2

    env_args = {'observables': MVC_OBSERVABLES,
                'reward_signal': RewardSignal.BLS,
                'extra_action': ExtraAction.NONE,
                'optimisation_target': OptimisationTarget.MIN_COVER,
                'spin_basis': SpinBasis.SIGNED,
                'norm_rewards': True,
                'memory_length': None,
                'horizon_length': None,
                'stag_punishment': None,
                'basin_reward': 1. / 60,
                'reversible_spins': True}

    ####################################################
    # LOAD VALIDATION GRAPHS
    ####################################################

    all_graphs = [load_graph_set(graph_save_loc), 
                  load_graph_set(graphs_40_loc),
                  load_graph_set(graphs_60_loc),
                  load_graph_set(graphs_80_loc),
                  load_graph_set(graphs_100_loc),]
    vert_counts = [20, 40, 60, 80, 100]

    ### CONVERT GRAPHS TO UNIFORM
    for diff_vert_count in all_graphs:
        for i in range(len(diff_vert_count)):
            diff_vert_count[i] = np.array(diff_vert_count[i] != 0, dtype=np.float64)
    
    
    ####################################################
    # SETUP NETWORK TO TEST
    ####################################################

    test_env = ising_env.make("SpinSystem",
                              SingleGraphGenerator(all_graphs[0][0]),
                              all_graphs[0][0].shape[0]*step_factor,
                              **env_args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.device(device)
    print("Set torch default device to {}.".format(device))

    network = network_fn(n_obs_in=test_env.observation_space.shape[1],
                         **network_args).to(device)

    network.load_state_dict(torch.load(network_save_path,map_location=device))
    for param in network.parameters():
        param.requires_grad = False
    network.eval()

    print("Sucessfully created agent with pre-trained MPNN.\nMPNN architecture\n\n{}".format(repr(network)))

    batch_size = 50
    cplex_solutions = []
    cover_matching_solutions = []
    greedy_start_solutions = []
    greedy_random_solutions = []
    networkx_solutions = []
    neural_network_empty_start_solutions = []
    neural_network_full_start_solutions = []
    neural_network_random_solutions = []

    test_envs : list[SpinSystemBase] = [None] * batch_size

    for i, graphs in enumerate(all_graphs):
        cplex_batch = []
        cover_matching_batch = []
        greedy_start_batch = []
        greedy_random_batch = []
        networkx_batch = []
        neural_network_empty_start_batch = []
        neural_network_full_start_batch = []
        neural_network_random_batch = []


        for j, test_graph in enumerate(graphs):
            env_args = {
                'observables': MVC_OBSERVABLES,
                'reward_signal': RewardSignal.BLS,
                'extra_action': ExtraAction.NONE,
                'optimisation_target': OptimisationTarget.MIN_COVER,
                'spin_basis': SpinBasis.SIGNED,
                'norm_rewards': True,
                'memory_length': None,
                'horizon_length': None,
                'stag_punishment': None,
                'basin_reward': 1. / test_graph.shape[0],
                'reversible_spins': True
            }
            
            # Create the environments that the solvers will run on
            # For solvers that only need to be tested once, just use the first environment
            print("Preparing batch of {} environments for graph {} with |V| = {}...".format(batch_size,j,test_graph.shape[0]))
            for i in range(batch_size):
                test_envs[i] = ising_env.make("SpinSystem",
                                            SingleGraphGenerator(test_graph),
                                            test_graph.shape[0]*step_factor,
                                            **env_args)
            

            cplex_solver = CplexSolver(env=test_envs[0])
            cplex_solver.reset()
            cplex_solver.solve()
            cplex_batch.append(test_envs[0].scorer.get_solution(test_envs[0].state[0, :test_envs[0].n_spins], test_envs[0].matrix))

            # Next test cover matching (run 50 tests on each graph)
            print("Running Matching Algorithm")
            matching_solutions_batch = []
            for i in range(batch_size):
                matching_solver = CoverMatching(env=test_envs[i])
                matching_solver.reset()
                matching_solver.solve()
                matching_solutions_batch.append(test_envs[i].scorer.get_solution(test_envs[i].state[0, :test_envs[i].n_spins], test_envs[i].matrix))

            cover_matching_batch.append(matching_solutions_batch)

            print("Running Greedy Algorithm from empty state")
            # Next test greedy allowing reversal from start state
            greedy_solver = Greedy(env=test_envs[0])
            greedy_solver.reset()
            greedy_solver.solve()
            greedy_start_batch.append(test_envs[0].best_solution)

            # Next test greedy allowing reversal from random state (run 50 tests on each graph)
            print("Running Greedy Algorithm from random state")
            greedy_random_sub_batch = []
            for i in range(batch_size):
                greedy_solver = Greedy(env=test_envs[i])
                greedy_solver.reset()
                greedy_solver.solve()
                greedy_random_sub_batch.append(test_envs[i].best_solution)

            greedy_random_batch.append(greedy_random_sub_batch)

            print("Running NetworkX min_weighted_cover algorithm")
            # Next test networkx minimum weighted cover
            networkx_solver = NetworkXMinCoverSolver(env=test_envs[0])
            networkx_solver.reset()
            networkx_solver.solve()
            networkx_batch.append(test_envs[i].scorer.get_solution(test_envs[i].state[0, :test_envs[i].n_spins], test_envs[0].matrix))

            # Next test network from empty state
            # First reset the environment to be empty, getting the observations
            print("Running GECO on empty initial state.")
            obs_batch = test_envs[0].reset([-1] * test_envs[0].n_spins)
            done = False
            while not done:
                obs_batch = torch.FloatTensor(np.array(obs_batch)).to(device)
                action = predict(network, obs_batch, test_envs[0].reversible_spins, test_envs[0].get_allowed_action_states())[0]
                obs, rew, done, info = test_envs[0].step(action)
                obs_batch = obs

            # Once done, get best solution found
            neural_network_empty_start_batch.append(test_envs[0].best_solution)

            # Next test network from full state
            print("Running GECO on full initial state")
            obs_batch = test_envs[0].reset([1] * test_envs[0].n_spins)
            done = False
            while not done:
                obs_batch = torch.FloatTensor(np.array(obs_batch)).to(device)
                action = predict(network, obs_batch, test_envs[0].reversible_spins, test_envs[0].get_allowed_action_states())[0]
                obs, rew, done, info = test_envs[0].step(action)
                obs_batch = obs

            # Once done, get best solution found
            neural_network_full_start_batch.append(test_envs[0].best_solution)
            
            # Next test network from random state (run 50 tests on each graph)
            print("Running GECO on random inital state.")
            obs_batch = []
            done = False
            # Reset all the environments
            for env in test_envs:
                obs_batch.append(env.reset())

            while not done:
                obs_batch = torch.FloatTensor(np.array(obs_batch)).to(device)
                # All envs in the batch have the same parameters, so pass them all to the predict
                actions = predict(network, obs_batch, test_envs[0].reversible_spins, test_envs[0].get_allowed_action_states())

                obs_batch = []
                for env, action in zip(test_envs, actions):
                    obs, rew, done, info = env.step(action)
                    obs_batch.append(obs)

            # Once done, add a list of every best solution to the solutions array
            neural_network_random_batch.append([env.best_solution for env in test_envs])

        cplex_solutions.append(cplex_batch)
        cover_matching_solutions.append(cover_matching_batch)
        greedy_start_solutions.append(greedy_start_batch)
        greedy_random_solutions.append(greedy_random_batch)
        networkx_solutions.append(networkx_batch)
        neural_network_empty_start_solutions.append(neural_network_empty_start_batch)
        neural_network_full_start_solutions.append(neural_network_full_start_batch)
        neural_network_random_solutions.append(neural_network_random_batch)

    """
    cplex_solutions = []
    cover_matching_solutions = []
    greedy_start_solutions = []
    greedy_random_solutions = []
    networkx_solutions = []
    neural_network_empty_start_solutions = []
    neural_network_full_start_solutions = []
    neural_network_random_solutions = []
    """

    cover_matching_solutions_avg = []
    greedy_random_solutions_avg = []
    neural_network_random_solutions_avg = []

    # For every randomized algorithm, whether through start state or implementation, reduce each to the average of the batched solutions
    for i in range(len(all_graphs)):
        for j in range(len(all_graphs[i])):
            cover_matching_solutions_avg.append(np.average(cover_matching_solutions[i][j]))
            greedy_random_solutions_avg.append(np.average(greedy_random_solutions[i][j]))
            neural_network_random_solutions_avg.append(np.average(neural_network_random_solutions[i][j]))

    solution_data = [[np.average(x) for x in cplex_solutions], 
                     [np.average(x) for x in cover_matching_solutions], 
                     [np.average(x) for x in greedy_start_solutions],
                     [np.average(x) for x in greedy_random_solutions],
                     [np.average(x) for x in networkx_solutions],
                     [np.average(x) for x in neural_network_empty_start_solutions],
                     [np.average(x) for x in neural_network_full_start_solutions],
                     [np.average(x) for x in neural_network_random_solutions]]
    
    # Print this data to file in case the graphs are messed up
    with open("test_data60.txt", 'w') as f:
        f.write(str(solution_data))


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


if __name__ == "__main__":
    run()