import os
import json
import matplotlib.pyplot as plt
import torch
import numpy as np
import time

import src.envs.core as ising_env
from experiments.utils import test_network, load_graph_set
from src.envs.utils import (SingleGraphGenerator,
                            RewardSignal, ExtraAction,
                            OptimisationTarget, SpinBasis,
                            EdgeType, Observable, DEFAULT_OBSERVABLES, MAIN_OBSERVABLES)
from src.networks.mpnn import MPNN
from src.agents.solver import *
from src.envs.spinsystem import SpinSystemBase

try:
    import seaborn as sns
    plt.style.use('seaborn')
except ImportError:
    pass

def run(num_vertices, problem_type, graph_type, problem_params, fixed_algorithms : list[SpinSolver], random_algorithms : list[SpinSolver], stepped_algorithms : list[SpinSolver]):
    
    save_loc = '{}_{}spin/eco/{}'.format(graph_type, num_vertices, problem_type)

    print("\n----- Running {} -----\n".format(os.path.basename(__file__)))

    ####################################################
    # NETWORK LOCATION
    ####################################################

    data_folder = os.path.join(save_loc, 'data')
    network_folder = os.path.join(save_loc, 'network')

    print("data folder :", data_folder)
    print("network folder :", network_folder)


    old_data = {}
    old_times = {}

    if os.path.exists("data/{}_test_data{}.txt".format(problem_type, num_vertices)):
        with open("data/{}_test_data{}.txt".format(problem_type, num_vertices), 'r') as f:
            old_data = json.loads(f.read().replace("'", '"'))
    
    if os.path.exists("data/{}_test_times{}.txt".format(problem_type, num_vertices)):
        with open("data/{}_test_times{}.txt".format(problem_type, num_vertices), 'r') as f:
            old_times = json.loads(f.read().replace("'", '"'))
    
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

    env_args = {'observables':problem_params['observables'],
                'reward_signal':problem_params['reward_signal'],
                'extra_action':ExtraAction.NONE,
                'optimisation_target':problem_params['optimisation'],
                'spin_basis':SpinBasis.SIGNED,
                'norm_rewards':True,
                'memory_length':None,
                'horizon_length':None,
                'stag_punishment':None,
                'basin_reward':problem_params['basin_reward'],
                'reversible_spins':problem_params['reversible_spins']}

    ####################################################
    # LOAD VALIDATION GRAPHS
    ####################################################

    graph_sizes = [20, 40, 60, 80, 100, 200, 500]
    all_graphs = []

    ### TODO modify this to work with BA graphs as well
    for i in graph_sizes:
        all_graphs.append(load_graph_set("_graphs/validation/{}_{}spin_p15_100graphs.pkl".format(graph_type, i)))

    ### CONVERT GRAPHS TO UNIFORM
    if problem_params['edge_type'] == EdgeType.UNIFORM:
        for diff_vert_count in all_graphs:
            for i in range(len(diff_vert_count)):
                diff_vert_count[i] = np.array(diff_vert_count[i] != 0, dtype=np.float64)

# MAX DOM SET
# CLIQUE
    
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

    ### SETUP TESTS
    ### Forcing cplex to be the first because it's fixed
    solutions = {}
    times = {}
    histories = {}

    batch_size = 50
    for algorithm in fixed_algorithms + random_algorithms:
        solutions['{}'.format(algorithm.name)] = {}
        times['{}'.format(algorithm.name)] = {}

    for algorithm in stepped_algorithms:
        solutions['{} empty start'.format(algorithm.name)] = {}
        solutions['{} full start'.format(algorithm.name)] = {} 
        times['{} empty start'.format(algorithm.name)] = {}
        times['{} full start'.format(algorithm.name)] = {}

    # solutions['neural network empty {}'.format(num_vertices)] = {}
    # solutions['neural network full {}'.format(num_vertices)] = {}
    # solutions['neural network random {}'.format(num_vertices)] = {}
    # times['neural network empty {}'.format(num_vertices)] = {}
    # times['neural network full {}'.format(num_vertices)] = {}
    # times['neural network random {}'.format(num_vertices)] = {}
    solutions['neural network partial {}'.format(num_vertices)] = {}
    times['neural network partial {}'.format(num_vertices)] = {}
    histories['neural network partial {}'.format(num_vertices)] = {}


    test_envs : list[SpinSystemBase] = [None] * batch_size

    for i, graphs in enumerate(all_graphs):
        # append batches
        for algorithm in solutions:
            solutions[algorithm][str(graphs[0].shape[0])] = []
            times[algorithm][str(graphs[0].shape[0])] = []

            if algorithm in old_data and str(graphs[0].shape[0]) in old_data[algorithm]:
                solutions[algorithm][str(graphs[0].shape[0])] = old_data[algorithm][str(graphs[0].shape[0])]
            if algorithm in old_times and str(graphs[0].shape[0]) in old_data[algorithm]:
                times[algorithm][str(graphs[0].shape[0])] = old_times[algorithm][str(graphs[0].shape[0])]

        histories['neural network partial {}'.format(num_vertices)][str(graphs[0].shape[0])] = []

        for j, test_graph in enumerate(graphs):
            env_args = {
                'observables': problem_params['observables'],
                'reward_signal': problem_params['reward_signal'],
                'extra_action': ExtraAction.NONE,
                'optimisation_target': problem_params['optimisation'],
                'spin_basis': SpinBasis.SIGNED,
                'norm_rewards': True,
                'memory_length': None,
                'horizon_length': None,
                'stag_punishment': None,
                'basin_reward': 1. / test_graph.shape[0],
                'reversible_spins': problem_params['reversible_spins']
            }
            
            # Create the environments that the solvers will run on
            # For solvers that only need to be tested once, just use the first environment
            print("Preparing batch of {} environments for graph {} with |V| = {}...".format(batch_size,j,test_graph.shape[0]))
            for i in range(batch_size):
                test_envs[i] = ising_env.make("SpinSystem",
                                            SingleGraphGenerator(test_graph),
                                            test_graph.shape[0]*step_factor,
                                            **env_args)
            
            for algorithm in fixed_algorithms:
                if algorithm.name not in old_data or str(test_graph.shape[0]) not in old_data[algorithm.name]:
                    algorithm.set_env(test_envs[0])
                    algorithm.reset()
                    start = time.time()
                    algorithm.solve()
                    end = time.time()
                    solutions[algorithm.name][str(test_graph.shape[0])].append(algorithm.measure)
                    times[algorithm.name][str(test_graph.shape[0])].append(end - start)
                else:
                    print("Previously computed {} based on old data, skipping.".format(algorithm.name))


            for algorithm in stepped_algorithms:
                if algorithm.name not in old_data or str(test_graph.shape[0]) not in old_data[algorithm.name]:
                    # empty start
                    algorithm.set_env(test_envs[0])
                    algorithm.reset([-1] * test_envs[0].n_spins)
                    start = time.time()
                    algorithm.solve()
                    end = time.time()

                    solutions['{} empty start'.format(algorithm.name)][str(test_graph.shape[0])].append(algorithm.measure)
                    times['{} empty start'.format(algorithm.name)][str(test_graph.shape[0])].append(end - start)
                    
                    algorithm.reset([1] * test_envs[0].n_spins)
                    start = time.time()
                    algorithm.solve()
                    end = time.time()

                    solutions['{} full start'.format(algorithm.name)][str(test_graph.shape[0])].append(algorithm.measure)
                    times['{} full start'.format(algorithm.name)][str(test_graph.shape[0])].append(end - start)
                else:
                    print("Previously computed {} based on old data, skipping.".format(algorithm.name))

            for algorithm in random_algorithms:
                if algorithm.name not in old_data or str(test_graph.shape[0]) not in old_data[algorithm.name]:
                    # setup sub-batches for randomized algorithms
                    solutions[algorithm.name][str(test_graph.shape[0])].append([])
                    times[algorithm.name][str(test_graph.shape[0])].append([])

                    for i in range(batch_size):
                        algorithm.set_env(test_envs[i])
                        algorithm.reset()
                        start = time.time()
                        algorithm.solve()
                        end = time.time()

                        solutions[algorithm.name][str(test_graph.shape[0])][-1].append(algorithm.measure)
                        times[algorithm.name][str(test_graph.shape[0])][-1].append(end - start)
                else:
                    print("Previously computed {} based on old data, skipping.".format(algorithm.name))

            # Next test network from random state (run 50 tests on each graph)
            # This is specifically for max_ind_set
            print("Running GECO on partial solution inital state.")
            # First thing, create partial solution for all environments
            # for env in test_envs:
            #     vertices = np.array([-1] * env.n_spins, dtype=np.float64)
            #     matrix = env.matrix
            #     # Make sure that the score mask has at least one positive value (indicating a vertex that can be added)
            #     while np.any(env.scorer.get_score_mask(vertices, matrix) > 0):
            #         # create all indexes of valid score increases
            #         indexes = [i for i in range(len(vertices)) if env.scorer.get_score_mask(vertices, matrix)[i] > 0]
            #         index = random.choice(indexes)
            #         vertices[index] = 1

            #     network_solver.reset(vertices)
            network_solver = Network(network=network, env=test_envs[0], name='network partial solution');

            start = time.time()
            vertices = np.array([-1] * test_envs[0].n_spins, dtype=np.float64)
            matrix = test_envs[0].matrix 

            while np.any(network_solver.env.scorer.get_score_mask(vertices, matrix) > 0):
                indexes = [i for i in range(len(vertices)) if network_solver.env.scorer.get_score_mask(vertices, matrix)[i] > 0]
                index = random.choice(indexes)
                vertices[index] = 1

            network_solver.reset(vertices)

            network_solver.solve()
            end = time.time()
            
            # Once done, get best solution found into the batch
            solutions['neural network partial {}'.format(num_vertices)][str(test_graph.shape[0])].append(test_envs[0].best_solution)
            times['neural network partial {}'.format(num_vertices)][str(test_graph.shape[0])].append(end - start)

            if len(histories['neural network partial {}'.format(num_vertices)][str(test_graph.shape[0])]) < 10:
                histories['neural network partial {}'.format(num_vertices)][str(test_graph.shape[0])].append(network_solver.history)

            # Next test network from empty state
            # First reset the environment to be empty, getting the observations
            # print("Running GECO on empty initial state.")
            # obs_batch = test_envs[0].reset([-1] * test_envs[0].n_spins)
            # start = time.time()
            # done = False
            # while not done:
            #     obs_batch = torch.FloatTensor(np.array(obs_batch)).to(device)
            #     action = predict(network, obs_batch, test_envs[0].reversible_spins, test_envs[0].get_allowed_action_states())[0]
            #     obs, rew, done, info = test_envs[0].step(action)
            #     obs_batch = obs
            
            # end = time.time()

            # # Once done, get best solution found into the batch
            # solutions['neural network empty {}'.format(num_vertices)][str(test_graph.shape[0])].append(test_envs[0].best_solution)
            # times['neural network empty {}'.format(num_vertices)][str(test_graph.shape[0])].append(end - start)

            # # Next test network from full state
            # print("Running GECO on full initial state")
            # obs_batch = test_envs[0].reset([1] * test_envs[0].n_spins)
            # start = time.time()
            # done = False
            # while not done:
            #     obs_batch = torch.FloatTensor(np.array(obs_batch)).to(device)
            #     action = predict(network, obs_batch, test_envs[0].reversible_spins, test_envs[0].get_allowed_action_states())[0]
            #     obs, rew, done, info = test_envs[0].step(action)
            #     obs_batch = obs

            # end = time.time()

            # # Once done, get best solution found
            # solutions['neural network full {}'.format(num_vertices)][str(test_graph.shape[0])].append(test_envs[0].best_solution)
            # times['neural network full {}'.format(num_vertices)][str(test_graph.shape[0])].append(end - start)
            
            # # Next test network from random state (run 50 tests on each graph)
            # print("Running GECO on random inital state.")
            # obs_batch = []
            # done = False
            # # Reset all the environments
            # for env in test_envs:
            #     obs_batch.append(env.reset())

            # start = time.time()

            # while not done:
            #     obs_batch = torch.FloatTensor(np.array(obs_batch)).to(device)
            #     # All envs in the batch have the same parameters, so pass them all to the predict
            #     actions = predict(network, obs_batch, test_envs[0].reversible_spins, test_envs[0].get_allowed_action_states())

            #     obs_batch = []
            #     for env, action in zip(test_envs, actions):
            #         obs, rew, done, info = env.step(action)
            #         obs_batch.append(obs)

            # end = time.time()

            # # Once done, add a list of every best solution to the solutions array
            # solutions['neural network random {}'.format(num_vertices)][str(test_graph.shape[0])].append([env.best_solution for env in test_envs])
            # times['neural network random {}'.format(num_vertices)][str(test_graph.shape[0])].append(end - start)
    
        # Print this data to file for every new batch to save partway through
        with open("data/{}_test_data{}.txt".format(problem_type, num_vertices), 'w') as f:
            f.write(str(solutions))

        with open("data/{}_test_times{}.txt".format(problem_type, num_vertices), 'w') as f:
            f.write(str(times))

        with open("data/{}_histories{}.txt".format(problem_type, num_vertices), 'w') as f:
            f.write(str(histories))


def run_with_params(num_vertices : int = 20, problem_type : str = 'min_cover', graph_type : str = 'ER', network_type='eco'):
    if problem_type == 'min_cover':
        problem_params = {
            'optimisation': OptimisationTarget.MIN_COVER,
            'edge_type': EdgeType.UNIFORM,
            'observables': MAIN_OBSERVABLES,
            'reversible_spins': True,
            'basin_reward': 1./num_vertices,
            'reward_signal': RewardSignal.BLS
        }
        fixed_algorithms = [CplexSolver(env=None, name='cplex'), NetworkXMinCoverSolver(env=None, name='networkx')]
        stepped_algorithms = [Greedy(env=None, name='greedy')]
        random_algorithms = [Greedy(env=None, name='greedy random'), CoverMatching(env=None, name='matching')]
    elif problem_type == 'max_cut':
        problem_params = {
            'optimisation': OptimisationTarget.CUT,
            'edge_type': EdgeType.DISCRETE,
            'observables': DEFAULT_OBSERVABLES,
            'reversible_spins': True,
            'basin_reward': 1./num_vertices,
            'reward_signal': RewardSignal.BLS
        }
        fixed_algorithms = []
        stepped_algorithms = []
        random_algorithms = []
    elif problem_type == 'min_cut':
        problem_params = {
            'optimisation': OptimisationTarget.MIN_CUT,
            'edge_type': EdgeType.DISCRETE,
            'observables': DEFAULT_OBSERVABLES,
            'reversible_spins': True,
            'basin_reward': 1./num_vertices,
            'reward_signal': RewardSignal.BLS
        }
        fixed_algorithms = []
        stepped_algorithms = []
        random_algorithms = []
    elif problem_type == 'max_ind_set':
        problem_params = {
            'optimisation': OptimisationTarget.MAX_IND_SET,
            'edge_type': EdgeType.UNIFORM,
            'observables': MAIN_OBSERVABLES,
            'reversible_spins': True,
            'basin_reward': 1./num_vertices,
            'reward_signal': RewardSignal.BLS
        }
        # Not doing CPLEX yet for time purposes
        fixed_algorithms = [NetworkXMaxIndSetSolver(env = None, name = 'networkx')]
        stepped_algorithms = []
        random_algorithms = []
    else:
        print('Invalid problem type.')
        exit(1)

    if network_type == 'eco':
        pass
    elif network_type == 's2v':
        problem_params['observables'] = [Observable.SPIN_STATE]
        problem_params['reversible_spins'] = False
        problem_params['basin_reward'] = None
        problem_params['reward_signal'] = RewardSignal.DENSE
    else:
        print('Invalid network type.')
        exit(1)

    run(num_vertices, problem_type, graph_type, problem_params, fixed_algorithms, random_algorithms, stepped_algorithms)


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
    num_vertices = 20
    graph_type = 'ER'
    problem_type = 'min_cover'
    network_type = 'eco'
    run_with_params(num_vertices, problem_type, graph_type, network_type)
