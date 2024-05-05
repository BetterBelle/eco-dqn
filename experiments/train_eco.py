import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

import src.envs.core as ising_env
from experiments.utils import load_graph_set, mk_dir
from src.agents.dqn.dqn import DQN
from src.agents.dqn.utils import TestMetric
from src.envs.utils import (SetGraphGenerator,
                            RandomErdosRenyiGraphGenerator,
                            EdgeType, RewardSignal, ExtraAction,
                            OptimisationTarget, SpinBasis, 
                            DEFAULT_OBSERVABLES, MAIN_OBSERVABLES,
                            RandomBarabasiAlbertGraphGenerator,
                            Observable, Stopping)
from src.networks.mpnn import MPNN

try:
    import seaborn as sns
    plt.style.use('seaborn')
except ImportError:
    pass

import time

def run(num_vertices, problem_type, dqn_params, problem_params, graph_type, train_graph_generator):
    save_loc = "{}_{}spin/eco/{}".format(graph_type, num_vertices, problem_type)

    print("\n----- Running {} -----\n".format(os.path.basename(__file__)))

    ####################################################
    # SET UP ENVIRONMENTAL AND VARIABLES
    ####################################################

    gamma=0.95
    step_fact = 2

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
    # SET UP TRAINING AND TEST GRAPHS
    ####################################################

    ####
    # Pre-generated test graphs
    ####
    graph_save_loc = "_graphs/testing/{}_{}spin_p15_50graphs.pkl".format(graph_type, num_vertices)
    graphs_test = load_graph_set(graph_save_loc)
    n_tests = len(graphs_test)

    # If we want uniform edges... instead of generating new graphs, take the discrete test graphs and make them uniform
    if problem_params['edge_type'] == EdgeType.UNIFORM:
        for i in range(len(graphs_test)):
            graphs_test[i] = np.array(graphs_test[i] != 0, dtype=np.float64)

    test_graph_generator = SetGraphGenerator(graphs_test, ordered=True)

    ####################################################
    # SET UP TRAINING AND TEST ENVIRONMENTS
    ####################################################

    train_envs = [ising_env.make("SpinSystem",
                                 train_graph_generator,
                                 int(num_vertices*step_fact),
                                 **env_args)]


    n_spins_test = train_graph_generator.get().shape[0]
    test_envs = [ising_env.make("SpinSystem",
                                test_graph_generator,
                                int(n_spins_test*step_fact),
                                **env_args)]

    ####################################################
    # SET UP FOLDERS FOR SAVING DATA
    ####################################################

    data_folder = os.path.join(save_loc,'data')
    network_folder = os.path.join(save_loc, 'network')

    mk_dir(data_folder)
    mk_dir(network_folder)
    
    network_save_path = os.path.join(network_folder,'network.pth')
    test_save_path = os.path.join(network_folder,'test_scores.pkl')
    loss_save_path = os.path.join(network_folder, 'losses.pkl')
    solutions_save_path = os.path.join(network_folder, 'solution.pkl')

    ####################################################
    # SET UP AGENT
    ####################################################

    nb_steps = dqn_params['num_steps']

    network_fn = lambda: MPNN(n_obs_in=train_envs[0].observation_space.shape[1],
                              n_layers=3,
                              n_features=64,
                              n_hid_readout=[],
                              tied_weights=False)

    agent = DQN(train_envs,

                network_fn,

                init_network_params=None,
                init_weight_std=0.01,

                double_dqn=True,
                clip_Q_targets=False,

                replay_start_size=dqn_params['replay_start_size'],
                replay_buffer_size=dqn_params['replay_buffer_size'],  # 20000
                gamma=gamma,  # 1
                update_target_frequency=dqn_params['update_target_frequency'],  # 500

                update_learning_rate=False,
                initial_learning_rate=1e-4,
                peak_learning_rate=1e-4,
                peak_learning_rate_step=20000,
                final_learning_rate=1e-4,
                final_learning_rate_step=200000,

                update_frequency=32,  # 1
                minibatch_size=64,  # 128
                max_grad_norm=None,
                weight_decay=0,

                update_exploration=True,
                initial_exploration_rate=1,
                final_exploration_rate=0.05,  # 0.05
                final_exploration_step=dqn_params['final_exploration_step'],  # 40000

                adam_epsilon=1e-8,
                logging=False,
                loss="mse",

                save_network_frequency=dqn_params['save_network_frequency'],
                network_save_path=network_save_path,

                evaluate=True,
                test_envs=test_envs,
                test_episodes=n_tests,
                test_frequency=dqn_params['test_frequency'],  # 10000
                test_save_path=test_save_path,
                test_metric=TestMetric.BEST,

                seed=None
                )

    print("\n Created DQN agent with network:\n\n", agent.network)

    #############
    # TRAIN AGENT
    #############
    start = time.time()
    agent.learn(timesteps=nb_steps, verbose=True)
    print(time.time() - start)

    agent.save()

    ############
    # PLOT - solution curve
    ############
    data = pickle.load(open(solutions_save_path,'rb'))
    data = np.array(data)

    fig_fname = os.path.join(network_folder,"training_curve")

    plt.plot(data[:,0],data[:,1])
    plt.xlabel("Timestep")
    plt.ylabel("Mean Solution Quality")
    if agent.test_metric == TestMetric.FINAL:
       plt.ylabel("Mean final solution")
    if agent.test_metric==TestMetric.ENERGY_ERROR:
      plt.ylabel("Energy Error")
    elif agent.test_metric==TestMetric.CUMULATIVE_REWARD:
      plt.ylabel("Cumulative Reward")

    plt.savefig(fig_fname + ".png", bbox_inches='tight')
    plt.savefig(fig_fname + ".pdf", bbox_inches='tight')

    plt.clf()

    ############
    # PLOT - score curve
    ############
    data = pickle.load(open(test_save_path,'rb'))
    data = np.array(data)

    fig_fname = os.path.join(network_folder,"score_curve")

    plt.plot(data[:,0],data[:,1])
    plt.xlabel("Timestep")
    plt.ylabel("Mean score")
    if agent.test_metric == TestMetric.FINAL:
       plt.ylabel("Mean final score")
    if agent.test_metric==TestMetric.ENERGY_ERROR:
      plt.ylabel("Energy Error")
    elif agent.test_metric==TestMetric.CUMULATIVE_REWARD:
      plt.ylabel("Cumulative Reward")

    plt.savefig(fig_fname + ".png", bbox_inches='tight')
    plt.savefig(fig_fname + ".pdf", bbox_inches='tight')

    plt.clf()

    ############
    # PLOT - losses
    ############
    data = pickle.load(open(loss_save_path,'rb'))
    data = np.array(data)

    fig_fname = os.path.join(network_folder,"loss")

    N=50
    data_x = np.convolve(data[:,0], np.ones((N,))/N, mode='valid')
    data_y = np.convolve(data[:,1], np.ones((N,))/N, mode='valid')

    plt.plot(data_x,data_y)
    plt.xlabel("Timestep")
    plt.ylabel("Loss")

    plt.yscale("log")
    plt.grid(True)

    plt.savefig(fig_fname + ".png", bbox_inches='tight')
    plt.savefig(fig_fname + ".pdf", bbox_inches='tight')



def run_with_vars(num_vertices=20, problem_type='min_cover', graph_type='ER', network_type='eco'):
    if problem_type == 'min_cover':
        problem_params = {
            'optimisation': OptimisationTarget.MIN_COVER,
            'edge_type': EdgeType.UNIFORM,
            'observables': MAIN_OBSERVABLES,
            'reversible_spins': True,
            'basin_reward': 1./num_vertices,
            'reward_signal': RewardSignal.BLS,
            'stopping': Stopping.NORMAL
        }
    elif problem_type == 'max_cut':
        problem_params = {
            'optimisation': OptimisationTarget.CUT,
            'edge_type': EdgeType.DISCRETE,
            'observables': DEFAULT_OBSERVABLES,
            'reversible_spins': True,
            'basin_reward': 1./num_vertices,
            'reward_signal': RewardSignal.BLS,
            'stopping': Stopping.NORMAL
        }
    elif problem_type == 'max_clique':
        problem_params = {
            'optimisation': OptimisationTarget.MAX_CLIQUE,
            'edge_type': EdgeType.UNIFORM,
            'observables': MAIN_OBSERVABLES,
            'reversible_spins': True,
            'basin_reward': 1./num_vertices,
            'reward_signal': RewardSignal.BLS,
            'stopping': Stopping.NORMAL
        }
    elif problem_type == 'min_cut':
        problem_params = {
            'optimisation': OptimisationTarget.MIN_CUT,
            'edge_type': EdgeType.DISCRETE,
            'observables': DEFAULT_OBSERVABLES,
            'reversible_spins': True,
            'basin_reward': 1./num_vertices,
            'reward_signal': RewardSignal.BLS,
            'stopping': Stopping.NORMAL
        }
    elif problem_type == 'max_ind_set':
        problem_params = {
            'optimisation': OptimisationTarget.MAX_IND_SET,
            'edge_type': EdgeType.UNIFORM,
            'observables': MAIN_OBSERVABLES,
            'reversible_spins': True,
            'basin_reward': 1./num_vertices,
            'reward_signal': RewardSignal.BLS,
            'stopping': Stopping.NORMAL
        }
    elif problem_type == 'min_dom_set':
        problem_params = {
            'optimisation': OptimisationTarget.MIN_DOM_SET,
            'edge_type': EdgeType.UNIFORM,
            'observables': MAIN_OBSERVABLES,
            'reversible_spins': True,
            'basin_reward': 1./num_vertices,
            'reward_signal': RewardSignal.BLS,
            'stopping': Stopping.NORMAL
        }
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

    

    if graph_type == 'ER':
        graph_generator = RandomErdosRenyiGraphGenerator(
            num_vertices, 
            0.15, 
            problem_params['edge_type']
        )
    elif graph_type == 'BA':
        graph_generator = RandomBarabasiAlbertGraphGenerator(
           num_vertices,
           4,
           problem_params['edge_type']
        )
    else:
        print('Invalid graph generator type.')
        exit(1)

    if num_vertices == 20 or num_vertices == 40:
        dqn_params = {
            'num_steps': 2500000,
            'replay_start_size': 500,
            'replay_buffer_size': 5000,
            'update_target_frequency': 1000,
            'final_exploration_step': 150000,
            'save_network_frequency': 100000,
            'test_frequency': 10000
        }
    elif num_vertices == 60:
        dqn_params = {
            'num_steps': 5000000,
            'replay_start_size': 500,
            'replay_buffer_size': 5000,
            'update_target_frequency': 1000,
            'final_exploration_step': 300000,
            'save_network_frequency': 200000,
            'test_frequency': 20000
        }
    elif num_vertices == 100:
        dqn_params = {
            'num_steps': 8000000,
            'replay_start_size': 1500,
            'replay_buffer_size': 10000,
            'update_target_frequency': 2500,
            'final_exploration_step': 800000,
            'save_network_frequency': 400000,
            'test_frequency': 50000
        }
    elif num_vertices == 200:
        dqn_params = {
            'num_steps': 10000000,
            'replay_start_size': 3000,
            'replay_buffer_size': 15000,
            'update_target_frequency': 4000,
            'final_exploration_step': 800000,
            'save_network_frequency': 400000,
            'test_frequency': 50000
        }
    else:
        print('Undefined vertex number.')
        exit(1)

    run(
        num_vertices, 
        problem_type, 
        dqn_params, 
        problem_params, 
        graph_type,
        graph_generator
    )

if __name__ == "__main__":
    num_vertices = 20
    problem_type = 'min_cover'
    graph_type = 'ER'
    network_type = 'eco'
    run_with_vars(num_vertices, problem_type, graph_type, network_type)
