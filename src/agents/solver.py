from abc import ABC, abstractmethod

from src.envs.spinsystem import SpinSystemBase
from src.envs.utils import OptimisationTarget
import numpy as np
import networkx as nx
import torch
import random
from docplex.mp.model import Model

class SpinSolver(ABC):
    """Abstract base class for agents solving SpinSystem Ising problems."""

    def __init__(self, env : SpinSystemBase, record_cut=False, record_rewards=False, record_qs=False, verbose=False, name=None):
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
        self.name = name
        self.measure = 0

        self.total_reward = 0

    def reset(self, spins = None):
        self.total_reward = 0
        self.env.reset(spins)

    def set_env(self, env):
        self.env = env

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

        # When done set measure to last seen state
        # This is only used for algorithms that step until termination
        self.measure = self.env.scorer.get_solution(self.env.state[0, :self.env.n_spins], self.env.matrix)
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

    epsilon = 0

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

        self.record_solution = True
        self.record_qs = True
        self.record_rewards = True
        self.record_spins = True

        self.history = []

    def reset(self, spins=None, clear_history=True):
        if type(self.env) != SpinSystemBase:
            pass
        self.current_observation = self.env.reset(spins)
        self.current_observation = torch.FloatTensor(self.current_observation).to(self.device)
        self.total_reward = 0

        if clear_history:
            self.history = []
            if not self.record_solution and not self.record_rewards:
                record = [0]
            else:
                record = [float(0)]
                if self.record_solution:
                    record += [float(self.env.scorer.get_solution(self.env.state[0, :self.env.n_spins], self.env.matrix))]
                if self.record_rewards:
                    record += [float(0)]
                if self.record_qs:
                    record += [[0] * self.env.n_spins]
                if self.record_spins:
                    record += [list(self.env.state[0, :self.env.n_spins])]

            record += [list(self.env.scorer.get_score_mask(self.env.state[0, :self.env.n_spins], self.env.matrix))]
            record += [self.env.scorer.is_valid(self.env.state[0, :self.env.n_spins], self.env.matrix)]

            self.history.append(record)

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
            record = [float(action)]
            if self.record_solution:
                record += [float(self.env.scorer.get_solution(self.env.state[0, :self.env.n_spins], self.env.matrix))]
            if self.record_rewards:
                record += [float(reward)]
            if self.record_qs:
                record += [qs.tolist()]
            if self.record_spins:
                record += [list(self.env.state[0, :self.env.n_spins])]

        record += [list(self.env.scorer.get_score_mask(self.env.state[0, :self.env.n_spins], self.env.matrix))]
        record += [self.env.scorer.is_valid(self.env.state[0, :self.env.n_spins], self.env.matrix)]

        self.history.append(record)

        return reward, done


class CoverMatching(SpinSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, spins=None):
        """
        Reset with an empty solution
        """
        super().reset([-1] * self.env.n_spins)

    def step(self):
        rew = 0 ### Because the solve function needs this
        done = False
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._solver : Model = None

    def reset(self, spins=None):
        """
        Reset the environment and prepare the solver
        """
        super().reset(spins)
        ### Generate the problem parameters
        if self.env.optimisation_target == OptimisationTarget.MIN_COVER:
            self._solver = Model('Minimum Vertex Cover')
            variables = self._solver.integer_var_list(len(self.env.matrix), 0, 1, 'x')

            for i in range(len(self.env.matrix)):
                # Don't double count edges, so start from index i in the row
                for j in range(i, len(self.env.matrix[i])):
                    if self.env.matrix[i][j] == 1:
                        self._solver.add_constraint(variables[i] + variables[j] >= 1, 'x_{} + x_{} >= 1'.format(i, j))

            self._solver.minimize(self._solver.sum(variables))
            self.measure = len(self.env.matrix)

        if self.env.optimisation_target == OptimisationTarget.MAX_IND_SET:
            self._solver = Model('Maximum Independent Set')
            variables = self._solver.integer_var_list(len(self.env.matrix), 0, 1, 'x')
            for i in range(len(self.env.matrix)):
                for j in range(i, len(self.env.matrix[i])):
                    if self.env.matrix[i][j] == 1:
                        self._solver.add_constraint(variables[i] + variables[j] <= 1, 'x_{} + x_{} <= 1'.format(i, j))

            self._solver.maximize(self._solver.sum(variables))
            self.measure = 0

        if self.env.optimisation_target == OptimisationTarget.MIN_DOM_SET:
            self._solver = Model('Minimum Dominating Set')
            variables = self._solver.integer_var_list(len(self.env.matrix), 0, 1, 'x')
            for i in range(len(self.env.matrix)):
                neighbours = []
                for j in range(i, len(self.env.matrix[i])):
                    if self.env.matrix[i][j] == 1:
                        neighbours.append(variables[j])
                self._solver.add_constraint(variables[i] + self._solver.sum(neighbours) >= 1, 'x_{} + neighbours >= 1'.format(i))

            self._solver.minimize(self._solver.sum(variables))
            self.measure = len(self.env.matrix)

        if self.env.optimisation_target == OptimisationTarget.MAX_CLIQUE:
            self._solver = Model('Maximum Clique')
            variables = self._solver.integer_var_list(len(self.env.matrix), 0, 1, 'x')
            for i in range(len(self.env.matrix)):
                for j in range(i + 1, len(self.env.matrix[i])):
                    if self.env.matrix[i][j] == 0:
                        self._solver.add_constraint(variables[i] + variables[j] <= 1, 'x_{} + x_{} <= 1'.format(i, j))

            self._solver.maximize(self._solver.sum(variables))
            self.measure = 0

    def solve(self, *args):
        self._solver.solve()
        self._solver.print_information()
        print(self._solver.solution)

        self.measure = self._solver.objective_value

    def step(self, *args):
        """
        Don't need step for this. cplex just solves instantly
        """
        raise NotImplementedError
    
class NetworkXSolver(SpinSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def solve(self, *args):
        """
        I'm going to assume the solution given is valid. Therefore measure is simply the size of the solution.
        """
        solution = []
        if self.env.optimisation_target == OptimisationTarget.MIN_COVER:
            solution = list(nx.algorithms.approximation.min_weighted_vertex_cover(nx.Graph(self.env.matrix)))
        elif self.env.optimisation_target == OptimisationTarget.MAX_IND_SET:
            solution = list(nx.algorithms.approximation.maximum_independent_set(nx.Graph(self.env.matrix)))
        elif self.env.optimisation_target == OptimisationTarget.MIN_DOM_SET:
            solution = list(nx.algorithms.approximation.min_weighted_dominating_set(nx.Graph(self.env.matrix)))
        elif self.env.optimisation_target == OptimisationTarget.MAX_CLIQUE:
            solution = list(nx.algorithms.approximation.max_clique(nx.Graph(self.env.matrix)))
        else:
            print("Invalid problem type for NetworkX")
            exit(1)

        # Solution for this is all the chosen nodes, therefore the length is the measure
        self.measure = len(solution)
    
    def step(self, *args):
        """
        Unneeded for networkx because it solves instantly
        """
        raise NotImplementedError
    
