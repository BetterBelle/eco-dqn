from abc import ABC, abstractmethod

from src.envs.spinsystem import SpinSystemBase
from src.envs.utils import OptimisationTarget
import numpy as np
import networkx as nx
import torch
import random
import cplex

class SpinSolver(ABC):
    """Abstract base class for agents solving SpinSystem Ising problems."""

    def __init__(self, env : SpinSystemBase, record_cut=False, record_rewards=False, record_qs=False, verbose=False):
        """Base initialisation of a SpinSolver.

        Args:
            env (SpinSystem): The environment (an instance of SpinSystem) with
                which the agent interacts.
            verbose (bool, optional): The logging verbosity.

        Attributes:
            env (SpinSystem): The environment (an instance of SpinSystem) with
                which the agent interacts.
            verbose (bool): The logging verbosity.
            total_reward (float): The cumulative total reward received.
        """

        self.env = env
        self.verbose = verbose
        self.record_solution = record_cut
        self.record_rewards = record_rewards
        self.record_qs = record_qs

        self.total_reward = 0

    def reset(self, spins = None):
        self.total_reward = 0
        self.env.reset(spins)

    def solve(self, *args):
        """Solve the SpinSystem by flipping individual spins until termination.

        Args:
            *args: The arguments passed through to the 'step' method to take the
                next action.  The implementation of 'step' depedens on the
                solver instance used.

        Returns:
            (float): The cumulative total reward received.

        """

        done = False
        while not done:
            reward, done = self.step(*args)
            self.total_reward += reward
        return self.total_reward

    @abstractmethod
    def step(self, *args):
        """Take the next step (flip the next spin).

        The implementation of 'step' depedens on the
                solver instance used.

        Args:
            *args: The arguments passed through to the 'step' method to take the
                next action.  The implementation of 'step' depedens on the
                solver instance used.

        Raises:
            NotImplementedError: Every subclass of SpinSolver must implement the
                step method.
        """

        raise NotImplementedError()

class Greedy(SpinSolver):
    """A greedy solver for a SpinSystem."""

    def __init__(self, *args, **kwargs):
        """Initialise a greedy solver.

        Args:
            *args: Passed through to the SpinSolver constructor.

        Attributes:
            trial_env (SpinSystemMCTS): The environment with in the agent tests
                actions (a clone of self.env where the final actions are taken).
            current_snap: The current state of the environment.
        """

        super().__init__(*args, **kwargs)

    def step(self):
        """Take the action which maximises the immediate reward.

        Returns:
            reward (float): The reward recieved.
            done (bool): Whether the environment is in a terminal state after
                the action is taken.
        """
        rewards_available = self.env.scorer.get_score_mask(self.env.state[0, :self.env.n_spins], self.env.matrix)

        if self.env.reversible_spins:
            action = rewards_available.argmax()
        else:
            masked_rewards_avaialable = rewards_available.copy()
            np.putmask(masked_rewards_avaialable,
                       self.env.get_observation()[0, :] != self.env.get_allowed_action_states(),
                       np.finfo(np.float64).min)
            action = masked_rewards_avaialable.argmax()

        if rewards_available[action] < 0:
            action = None
            reward = 0
            done = True
        else:
            observation, reward, done, _ = self.env.step(action)

        return reward, done

class Random(SpinSolver):
    """A random solver for a SpinSystem."""

    def __init__(self, *args, **kwargs):
        """Initialise a random solver.

        Args:
            *args: Passed through to the SpinSolver constructor.

        Attributes:
            trial_env (SpinSystemMCTS): The environment with in the agent tests
                actions (a clone of self.env where the final actions are taken).
            current_snap: The current state of the environment.
        """
        super().__init__(*args, **kwargs)

    def step(self):
        """Take a random action.

        Returns:
            reward (float): The reward recieved.
            done (bool): Whether the environment is in a terminal state after
                the action is taken.
        """

        observation, reward, done, _ = self.env.step(self.env.action_space.sample())
        return reward, done

class Network(SpinSolver):
    """A network-only solver for a SpinSystem."""

    epsilon = 0.

    def __init__(self, network, *args, **kwargs):
        """Initialise a network-only solver.

        Args:
            network: The network.
            *args: Passed through to the SpinSolver constructor.

        Attributes:
            current_snap: The last observation of the environment, used to choose the next action.
        """

        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = network.to(self.device)
        self.network.eval()
        self.current_observation = self.env.get_observation()
        self.current_observation = torch.FloatTensor(self.current_observation).to(self.device)

        self.history = []

    def reset(self, spins=None, clear_history=True):
        self.current_observation = self.env.reset(spins)
        self.current_observation = torch.FloatTensor(self.current_observation).to(self.device)
        self.total_reward = 0

        if clear_history:
            self.history = []

    @torch.no_grad()
    def step(self):

        # Q-values predicted by the network.
        qs = self.network(self.current_observation)

        if self.env.reversible_spins:
            if np.random.uniform(0, 1) >= self.epsilon:
                # Action that maximises Q function
                action = qs.argmax().item()
            else:
                # Random action
                action = np.random.randint(0, self.env.action_space.n)

        else:
            x = (self.current_observation[0, :] == self.env.get_allowed_action_states()).nonzero()
            if np.random.uniform(0, 1) >= self.epsilon:
                action = x[qs[x].argmax().item()].item()
                # Allowed action that maximises Q function
            else:
                # Random allowed action
                action = x[np.random.randint(0, len(x))].item()

        if action is not None:
            observation, reward, done, _ = self.env.step(action)
            self.current_observation = torch.FloatTensor(observation).to(self.device)

        else:
            reward = 0
            done = True

        if not self.record_solution and not self.record_rewards:
            record = [action]
        else:
            record = [action]
            if self.record_solution:
                record += [self.env.scorer.get_solution(self.env.state[0, :self.env.n_spins], self.env.matrix)]
            if self.record_rewards:
                record += [reward]
            if self.record_qs:
                record += [qs]

        record += [self.env.scorer.get_score_mask(self.env.state[0, :self.env.n_spins], self.env.matrix)]

        self.history.append(record)

        return reward, done


class CoverMatching(SpinSolver):
    def __init__(self, env: SpinSystemBase, record_cut=False, record_rewards=False, record_qs=False, verbose=False):
        super().__init__(env, record_cut, record_rewards, record_qs, verbose)

    def reset(self, spins=None):
        """
        Reset with an empty solution
        """
        super().reset([-1] * self.env.n_spins)

    def step(self):
        rew = 0 ### Because the solve function needs this
        spins = self.env.state[0, :self.env.n_spins]
        matrix = self.env.matrix

        ### Matrix containing only the uncovered edges
        edges = matrix * np.array([spins == -1]) * np.array([spins == -1]).T

        ### Gets x and y coordinates (i.e. node index) of the nodes incident on uncovered edges
        coordinates = list(zip(*np.where(edges == 1)))

        ### Gets the two nodes of the chosen edge
        chosen_edge = random.choice(coordinates)
        ### Make the action for both nodes to add them to the solution
        for vertex in chosen_edge:
            observation, reward, done, _ = self.env.step(vertex)
            rew += reward

        ### Now if we have a valid solution, we're done
        new_spins = self.env.state[0, :self.env.n_spins]
        ### if every single entry is a 0
        if np.all(matrix * np.array([new_spins == -1]) * np.array([new_spins == -1]).T == 0):
            done = True

        return rew, done


class CplexSolver(SpinSolver):
    def __init__(self, env: SpinSystemBase, record_cut=False, record_rewards=False, record_qs=False, verbose=False):
        super().__init__(env, record_cut, record_rewards, record_qs, verbose)
        self._solver = cplex.Cplex()
        self._solver.set_log_stream(None)
        self._solver.set_results_stream(None)

    def reset(self, spins=None):
        """
        Reset the environment and prepare the solver
        """
        super().reset(spins)
        ### Generate the problem parameters
        if self.env.optimisation_target == OptimisationTarget.MIN_COVER:
            self._solver.set_problem_name("Minimum Vertex Cover")
            self._solver.set_problem_type(cplex.Cplex.problem_type.LP)
            self._solver.objective.set_sense(self._solver.objective.sense.minimize)
            
            names = [str(i) for i in range(self.env.n_spins)]
            w_obj = [1] * self.env.n_spins
            lb = [0] * self.env.n_spins
            ub = [1] * self.env.n_spins

            self._solver.variables.add(names=names, obj=w_obj, lb=lb, ub=ub)

            all_int = [(var, self._solver.variables.type.integer) for var in names]
            self._solver.variables.set_types(all_int)

            constraints = []
            # Go through each node and add a constraint for the edges in the upper triangular part of the adjacency matrix
            for i in range(len(self.env.matrix)):
                ### Start from i in each row so we don't double count edges
                for j in range(i, len(self.env.matrix)):
                    ### If there is an edges from i to j, add a constraint with their names and weight 1
                    if self.env.matrix[i][j] == 1:
                        constraints.append([[str(i),str(j)], [1, 1]])

            # The names of the constrains, just being the string i,j
            constraint_names = [",".join(x[0]) for x in constraints]
            # Each edges must have at least one vertex
            rhs = [1] * len(constraints)
            # Constraint is always a lower limit, therefore "G"
            constraint_senses = ["G"] * len(constraints)
            self._solver.linear_constraints.add(names=constraint_names, 
                                                 lin_expr=constraints, 
                                                 senses=constraint_senses,
                                                 rhs=rhs)

    def solve(self):
        self._solver.solve()
        print("CPLEX solution result is {}".format(self._solver.solution.get_status_string()))
        self.env.reset(2 * np.array(self._solver.solution.get_values(), dtype=np.int64) - 1)

    def step(self, *args):
        """
        Don't need step for this. cplex just solves instantly
        """
        return None
    
class NetworkXMinCoverSolver(SpinSolver):
    def __init__(self, env: SpinSystemBase, record_cut=False, record_rewards=False, record_qs=False, verbose=False):
        super().__init__(env, record_cut, record_rewards, record_qs, verbose)

    def reset(self):
        """
        Reset the environment and prepare the solver/solution
        """
        super().reset()
        self.env.reset([-1] * self.env.n_spins)
        self._solution = list(nx.algorithms.approximation.min_weighted_vertex_cover(nx.Graph(self.env.matrix)))
        self._current_index = 0

    def step(self):
        """
        Get the index of the current action to apply and do it on the environment
        """
        rew = 0
        action = self._solution[self._current_index]
        observation, reward, done, _ = self.env.step(action)

        rew += reward

        self._current_index += 1
        if self._current_index == len(self._solution):
            done = True

        return rew, done
        
