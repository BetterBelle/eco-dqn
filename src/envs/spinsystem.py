from collections import namedtuple
from networkx import is_valid_joint_degree

import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt

from src.envs.score_solver import ScoreSolverFactory
from src.envs.utils import (EdgeType,
                            RewardSignal,
                            ExtraAction,
                            OptimisationTarget,
                            Observable,
                            SpinBasis,
                            DEFAULT_OBSERVABLES,
                            GraphGenerator,
                            RandomGraphGenerator,
                            HistoryBuffer)

# A container for get_result function below. Works just like tuple, but prettier.
ActionResult = namedtuple("action_result", ("snapshot","observation","reward","is_done","info"))

class SpinSystemFactory(object):
    '''
    Factory class for returning new SpinSystem.
    '''

    @staticmethod
    def get(graph_generator=None,
            max_steps=20,
            observables = DEFAULT_OBSERVABLES,
            reward_signal = RewardSignal.DENSE,
            extra_action = ExtraAction.PASS,
            optimisation_target = OptimisationTarget.ENERGY,
            spin_basis = SpinBasis.SIGNED,
            norm_rewards=False,
            memory_length=None,  # None means an infinite memory.
            horizon_length=None, # None means an infinite horizon.
            stag_punishment=None, # None means no punishment for re-visiting states.
            basin_reward=None, # None means no reward for reaching a local minima.
            reversible_spins=True, # Whether the spins can be flipped more than once (i.e. True-->Georgian MDP).
            init_snap=None,
            seed=None):

        return SpinSystemBase(graph_generator, max_steps, observables, reward_signal, extra_action, optimisation_target, spin_basis,
                              norm_rewards,memory_length,horizon_length,stag_punishment,basin_reward,reversible_spins,init_snap,seed)

class SpinSystemBase():
    '''
    SpinSystemBase implements the functionality of a SpinSystem that is common to both
    biased and unbiased systems.  Methods that require significant enough changes between
    these two case to not readily be served by an 'if' statement are left abstract, to be
    implemented by a specialised subclass.
    '''

    # Note these are defined at the class level of SpinSystem to ensure that SpinSystem
    # can be pickled.
    class action_space():
        """
        The action space for a SpinSystem. This determines the number of actions and actions that can be done on the system.
        """
        def __init__(self, n_actions):
            self.n = n_actions
            self.actions = np.arange(self.n)

        def sample(self, n=1):
            """
            Chooses a random action from the action space
            """
            return np.random.choice(self.actions, n)

    class observation_space():
        """
        Represents the space of observations for the SpinSystem. This gives the number of vertices and observables in the system.
        """
        def __init__(self, n_spins, n_observables):
            self.shape = [n_spins, n_observables]

    def __init__(self,
                 graph_generator=None,
                 max_steps=20,
                 observables=DEFAULT_OBSERVABLES,
                 reward_signal = RewardSignal.DENSE,
                 extra_action = ExtraAction.PASS,
                 optimisation_target=OptimisationTarget.ENERGY,
                 spin_basis=SpinBasis.SIGNED,
                 norm_rewards=False,
                 memory_length=None,  # None means an infinite memory.
                 horizon_length=None,  # None means an infinite horizon.
                 stag_punishment=None,
                 basin_reward=None,
                 reversible_spins=False,
                 init_snap=None,
                 seed=None):
        '''
        Init method.

        Args:
            graph_generator: A GraphGenerator (or subclass thereof) object.
            max_steps: Maximum number of steps before termination.
            reward_signal: RewardSignal enum determining how and when rewards are returned.
            extra_action: ExtraAction enum determining if and what additional action is allowed,
                          beyond simply flipping spins.
            init_snap: Optional snapshot to load spin system into pre-configured state for MCTS.
            seed: Optional random seed.
        '''

        if seed != None:
            np.random.seed(seed)

        # Ensure first observable is the spin state.
        # This allows us to access the spins as self.state[0,:self.n_spins.]
        assert observables[0] == Observable.SPIN_STATE, "First observable must be Observation.SPIN_STATE."

        self.observables = list(enumerate(observables))

        self.extra_action = extra_action

        if graph_generator!=None:
            assert isinstance(graph_generator,GraphGenerator), "graph_generator must be a GraphGenerator implementation."
            self.gg = graph_generator
        else:
            # provide a default graph generator if one is not passed
            self.gg = RandomGraphGenerator(n_spins=20,
                                           edge_type=EdgeType.DISCRETE,
                                           biased=False,
                                           extra_action=(extra_action!=extra_action.NONE))

        self.n_spins = self.gg.n_spins  # Total number of spins in episode
        self.max_steps = max_steps  # Number of actions before reset

        self.reward_signal = reward_signal
        self.norm_rewards = norm_rewards

        self.n_actions = self.n_spins
        
        if extra_action != ExtraAction.NONE:
            self.n_actions+=1

        self.action_space = self.action_space(self.n_actions)
        self.observation_space = self.observation_space(self.n_spins, len(self.observables))

        self.current_step = 0

        if self.gg.biased:
            self.matrix, self.bias = self.gg.get()
        else:
            self.matrix = self.gg.get()
            self.bias = None

        self.optimisation_target = optimisation_target
        self.scorer = ScoreSolverFactory.get(optimisation_target, self.gg.biased)

        self.spin_basis = spin_basis

        self.memory_length = memory_length
        self.horizon_length = horizon_length if horizon_length is not None else self.max_steps
        self.stag_punishment = stag_punishment
        self.basin_reward = basin_reward
        self.reversible_spins = reversible_spins

        self.reset()

        self.score = self.scorer.get_score(self.state[0, :self.n_spins], self.matrix)
        self.normalized_score = self.scorer.get_normalized_score(self.state[0, :self.n_spins], self.matrix)

        if self.reward_signal == RewardSignal.SINGLE:
            self.init_score = self.score

        self.best_score = self.score
        self.best_spins = self.state[0,:]
        self.best_score_normalized = self.normalized_score

        if init_snap != None:
            self.load_snapshot(init_snap)

    def reset(self, spins=None):
        """
        Resets the SpinSystem variables to initial conditions, i.e. random spins, current step 0, setting up the GraphGenerator,
        resetting the graph observables, resetting the local rewards and so on.
        """
        self.current_step = 0

        if self.gg.biased:
            # self.matrix, self.bias = self.gg.get(with_padding=(self.extra_action != ExtraAction.NONE))
            self.matrix, self.bias = self.gg.get()
        else:
            # self.matrix = self.gg.get(with_padding=(self.extra_action != ExtraAction.NONE))
            self.matrix = self.gg.get()
        self._reset_graph_observables()

        # Generates an array of 1s of size number of spins
        empty_solution = np.array([-1] * self.n_spins, dtype=np.float64)

        # Gets the array saying how much reward you get from flipping each vertex (i.e. the solution improvement)
        local_rewards_available = self.scorer.get_score_mask(empty_solution, self.matrix)

        # Removes all zero values from that list
        local_rewards_available = local_rewards_available[np.nonzero(local_rewards_available)]

        ### Because we have an empty solution, having no positive or negative rewards available means we have an invalid graph
        if local_rewards_available.size == 0:
            # We've generated an empty graph, this is pointless, try again.
            self.reset()
        else:
            self.scorer.set_max_local_reward(empty_solution, self.matrix)

        ### Reset the state, does not include things relative to the best state because it's not yet set and the best state is the current one
        self.state = self._reset_state(spins)

        # Now we have a state and we know the graph is valid, set the normalizers for the scorer
        self.scorer.set_invalidity_normalizer(self.state[0, :self.n_spins], self.matrix)
        self.scorer.set_quality_normalizer(self.state[0, :self.n_spins], self.matrix)
        self.scorer.set_lower_bound(self.state[0, :self.n_spins], self.matrix)

        # Set the score and normalized score
        self.score = self.scorer.get_score(self.state[0, :self.n_spins], self.matrix)
        self.normalized_score = self.scorer.get_normalized_score(self.state[0, :self.n_spins], self.matrix)
        self.solution = self.scorer.get_solution(self.state[0, :self.n_spins], self.matrix)

        # nx.draw_networkx(nx.Graph(self.matrix))
        # plt.show()

        if self.reward_signal == RewardSignal.SINGLE:
            self.init_score = self.score
            self.init_score_normalized = self.normalized_score

        ### When resetting, the best score and best observed are just the current score
        self.best_score = self.score
        self.best_score_normalized = self.normalized_score
        self.best_obs_score = self.score
        self.best_obs_score_normalized = self.normalized_score


        ### Best solution is the actual solution of the current state, not the score
        self.best_solution = self.solution

        ### Best spins is of course state[0, :self.n_spins] because that holds all the spin booleans similar to score
        self.best_spins = self.state[0, :self.n_spins].copy()
        self.best_obs_spins = self.state[0, :self.n_spins].copy()

        ### Some memory setting that I don't need to worry about because it's just storing scores
        if self.memory_length is not None:
            self.score_memory = np.array([self.best_score] * self.memory_length)
            self.spins_memory = np.array([self.best_spins] * self.memory_length)
            self.idx_memory = 1

        self._reset_graph_observables()

        ### Keeps track of rewards for local minimums and punishments for stagnation
        if self.stag_punishment is not None or self.basin_reward is not None:
            self.history_buffer = HistoryBuffer()

        return self.get_observation()

    def _reset_graph_observables(self):
        """
        Resets the matrix of observations to be the same as the adjacency matrix, adding in case of extra actions and changing if the SpinSystem
        is biased (i.e. a directed graph)
        """
        # Reset observed adjacency matrix
        if self.extra_action != self.extra_action.NONE:
            # Pad adjacency matrix for disconnected extra-action spins of value 0.
            self.matrix_obs = np.zeros((self.matrix.shape[0] + 1, self.matrix.shape[0] + 1))
            self.matrix_obs [:-1, :-1] = self.matrix
        else:
            
            self.matrix_obs = self.matrix

        # Reset observed bias vector,
        if self.gg.biased:
            if self.extra_action != self.extra_action.NONE:
                # Pad bias for disconnected extra-action spins of value 0.
                self.bias_obs = np.concatenate((self.bias, [0]))
            else:
                self.bias_obs = self.bias

    def _reset_state(self, spins=None):
        """
        The state's immediate reward and greedy actions get reset based on current spins selected
        Note that for unreversible spins (i.e. S2V) all spins are set to -1 and can be flipped to 1 but for reversibles
        the spins get set to a random value either 1 or -1. If a spin list is passed in, we format them to be signed.
        """
        state = np.zeros((self.observation_space.shape[1], self.n_actions))

        if spins is None:
            if self.reversible_spins:
                # For reversible spins, initialise randomly to {+1,-1}.
                state[0, :self.n_spins] = 2 * np.random.randint(2, size=self.n_spins) - 1
            else:
                # For irreversible spins, initialise all to -1 (i.e. allowed to be flipped).
                state[0, :self.n_spins] = -1
        else:
            state[0, :] = self._format_spins_to_signed(spins)

        state = state.astype('float')

        immediate_quality_changes = self.scorer.get_solution_quality_mask(state[0, :self.n_spins], self.matrix)
        immediate_invalidity_changes = self.scorer.get_invalidity_degree_mask(state[0, :self.n_spins], self.matrix)

        # If any observables other than "immediate energy available" require setting to values other than
        # 0 at this stage, we should use a 'for k,v in enumerate(self.observables)' loop.
        for idx, obs in self.observables:

            ### Local observables ###
            if obs==Observable.IMMEDIATE_QUALITY_CHANGE:
                state[idx, :self.n_spins] = immediate_quality_changes / self.scorer._max_local_reward

            elif obs==Observable.IMMEDIATE_VALIDITY_DIFFERENCE:
                state[idx, :self.n_spins] = immediate_invalidity_changes / self.scorer._invalidity_normalizer

            elif obs==Observable.IMMEDIATE_VALIDITY_CHANGE:
                state[idx, :self.n_spins] = self.scorer.get_validity_mask(state[0, :self.n_spins], self.matrix) 

            ### Global observables ###
            elif obs==Observable.NUMBER_OF_QUALITY_IMPROVEMENTS:
                state[idx, :] = np.sum(immediate_quality_changes > 0) / self.n_spins

            elif obs==Observable.NUMBER_OF_VALIDITY_IMPROVEMENTS:
                state[idx, :] = np.sum(immediate_invalidity_changes > 0) / self.n_spins

            elif obs==Observable.VALIDITY_BIT:
                state[idx, :] = self.scorer.is_valid(state[0, :self.n_spins], self.matrix)

        return state

    def _get_spins(self, basis=SpinBasis.SIGNED):
        ### State[0, :self.n_spins] seems to be a boolean mask for chosen vertices
        ### Full state appears to be a list of the observations by index (0 is for spin state therefore spins gets that)
        spins = self.state[0, :self.n_spins]

        if basis == SpinBasis.SIGNED:
            pass
        ### They had this as SpinSystemBiased which makes no sense so I changed it, should be fine
        elif basis == SpinBasis.BINARY:
            # convert {1,-1} --> {0,1}
            spins[0, :] = (1 - spins[0, :]) / 2
        else:
            raise NotImplementedError("Unrecognised SpinBasis")

        return spins

    def seed(self, seed):
        return self.seed

    def set_seed(self, seed):
        self.seed = seed
        np.random.seed(seed)

    def step(self, action):
        done = False

        rew = 0 # Default reward to zero.
        randomised_spins = False

        ### Only increment the step if a valid solution has been found before
        self.current_step += 1

        if self.current_step > self.max_steps:
            print("The environment has already returned done. Stop it!")
            raise NotImplementedError

        new_state = np.copy(self.state)

        ############################################################
        # 1. Performs the action and calculates the score change. #
        ############################################################

        if action==self.n_spins:
            if self.extra_action == ExtraAction.PASS:
                delta_score = 0
            if self.extra_action == ExtraAction.RANDOMISE:
                # Randomise the spin configuration.
                randomised_spins = True
                random_actions = np.random.choice([1, -1], self.n_spins)
                new_state[0, :] = self.state[0, :] * random_actions
                new_score = self.scorer.get_score(new_state[0, :self.n_spins], self.matrix)
                new_score_normalized = self.scorer.get_normalized_score(new_state[0, :self.n_spins], self.matrix)

                delta_score = new_score - self.score
                delta_score_normalized = new_score_normalized - self.normalized_score

                self.score = new_score
                self.normalized_score = new_score_normalized
        else:
            # Calculate score change, which is just the index of the action for the current spins
            delta_score = self.scorer.get_score_mask(self.state[0, :], self.matrix)[action]
            delta_score_normalized = self.scorer.get_normalized_score_mask(self.state[0, :], self.matrix)[action]

            # Now change the state
            new_state[0,action] = -self.state[0,action]

            self.score += delta_score
            self.normalized_score += delta_score_normalized

        #############################################################################################
        # 2. Calculate reward for action and update anymemory buffers.                              #
        #   a) Calculate reward (always w.r.t best observable score).                              #
        #   b) If new global best has been found: update best ever score and spin parameters.      #
        #   c) If the memory buffer is finite (i.e. self.memory_length is not None):                #
        #          - Add score/spins to their respective buffers.                                  #
        #          - Update best observable score and spins w.r.t. the new buffers.                #
        #      else (if the memory is infinite):                                                    #
        #          - If new best has been found: update best observable score and spin parameters. #                                                                        #
        #############################################################################################

        self.state = new_state
        immediate_quality_changes = self.scorer.get_solution_quality_mask(self.state[0, :self.n_spins], self.matrix)
        immediate_invalidity_changes = self.scorer.get_invalidity_degree_mask(self.state[0, :self.n_spins], self.matrix)
        immediate_score_changes = self.scorer.get_score_mask(self.state[0, :self.n_spins], self.matrix)

        if self.score > self.best_obs_score:
            if self.reward_signal == RewardSignal.BLS:
                if self.norm_rewards:
                    rew = self.normalized_score - self.best_obs_score_normalized
                else:
                    rew = self.score - self.best_obs_score

            ### Don't know what this does as it isn't described in the paper, will ignore for now
            # elif self.reward_signal == RewardSignal.CUSTOM_BLS:
            #     rew = self.score - self.best_obs_score
            #     rew = rew / (rew + 0.1)
            #     if self.norm_rewards:
            #         rew /= self.n_spins
            ###

        ### Dense reward signal is for s2v-dqn
        if self.reward_signal == RewardSignal.DENSE:
            rew = delta_score_normalized if self.norm_rewards else delta_score
        # elif self.reward_signal == RewardSignal.SINGLE and done:
        #     rew = self.score - self.init_score
        ###
        # nx.draw_networkx(nx.Graph(self.matrix))
        # plt.show()

        if self.stag_punishment is not None or self.basin_reward is not None:
            visiting_new_state = self.history_buffer.update(action)

        if self.stag_punishment is not None:
            if not visiting_new_state:
                rew -= self.stag_punishment

        if self.basin_reward is not None:
            if np.all(immediate_score_changes <= 0):
                # All immediate score changes are negative <--> we are in a local optimum.
                if visiting_new_state:
                    # #####TEMP####
                    # if self.reward_signal != RewardSignal.BLS or (self.score > self.best_obs_score):
                    # ####TEMP####
                    rew += self.basin_reward

        if self.score > self.best_score:
            self.best_score = self.score
            self.best_score_normalized = self.normalized_score
            self.best_spins = self.state[0, :self.n_spins].copy()
            self.best_solution = self.scorer.get_solution(self.best_spins, self.matrix)

            # For the timed version, we'll also set the step back to zero
            self.current_step = 0

        if self.memory_length is not None:
            # For case of finite memory length.
            ### TODO: Figure out how this works and maybe consider adding normalized score
            self.score_memory[self.idx_memory] = self.score
            self.spins_memory[self.idx_memory] = self.state[0, :self.n_spins]
            self.idx_memory = (self.idx_memory + 1) % self.memory_length
            self.best_obs_score = self.score_memory.max()
            self.best_obs_spins = self.spins_memory[self.score_memory.argmax()].copy()
        else:
            ### TODO: add the normalized version of the score in here as well, so we don't have to recompute it every time
            self.best_obs_score = self.best_score
            self.best_obs_score_normalized = self.best_score_normalized
            self.best_obs_spins = self.best_spins.copy()

        #############################################################################################
        # 3. Updates the state of the system (except self.state[0,:] as this is always the spin     #
        #    configuration and has already been done.                                               #
        #   a) Update self.state local features to reflect the chosen action.                       #                                                                  #
        #   b) Update global features in self.state (always w.r.t. best observable score/spins)     #
        #############################################################################################

        for idx, observable in self.observables:

            ### Local observables ###
            if observable==Observable.IMMEDIATE_QUALITY_CHANGE:
                self.state[idx, :self.n_spins] = immediate_quality_changes / self.scorer._max_local_reward

            elif observable==Observable.TIME_SINCE_FLIP:
                self.state[idx, :] += (1. / self.max_steps)
                if randomised_spins:
                    self.state[idx, :] = self.state[idx, :] * (random_actions > 0)
                else:
                    self.state[idx, action] = 0

            elif observable==Observable.IMMEDIATE_VALIDITY_DIFFERENCE:
                self.state[idx, :self.n_spins] = immediate_invalidity_changes / self.scorer._invalidity_normalizer

            elif observable==Observable.IMMEDIATE_VALIDITY_CHANGE:
                self.state[idx, :self.n_spins] = self.scorer.get_validity_mask(self.state[0, :self.n_spins], self.matrix) 

            ### Global observables ###
            elif observable==Observable.EPISODE_TIME:
                self.state[idx, :] += (1. / self.max_steps)

            elif observable==Observable.TERMINATION_IMMANENCY:
                # Update 'Immanency of episode termination'
                self.state[idx, :] = max(0, ((self.current_step - self.max_steps) / self.horizon_length) + 1)

            elif observable==Observable.NUMBER_OF_QUALITY_IMPROVEMENTS:
                self.state[idx, :] = np.sum(immediate_quality_changes > 0) / self.n_spins

            elif observable==Observable.DISTANCE_FROM_BEST_SOLUTION:
                current_quality = self.scorer.get_solution_quality(self.state[0, :self.n_spins], self.matrix)
                best_quality = self.scorer.get_solution_quality(self.best_spins[:self.n_spins], self.matrix)
                self.state[idx, :] = np.abs(current_quality - best_quality) / self.scorer._max_local_reward

            elif observable==Observable.NUMBER_OF_VALIDITY_IMPROVEMENTS:
                # Immediate invalidity is negative when getting closer to a valid solution
                # Therefore we want the number of negative values.
                self.state[idx, :] = np.sum(immediate_invalidity_changes < 0) / self.n_spins

            elif observable==Observable.DISTANCE_FROM_BEST_STATE:
                self.state[idx, :] = np.count_nonzero(self.best_obs_spins[:self.n_spins] - self.state[0, :self.n_spins])

            elif observable==Observable.GLOBAL_VALIDITY_DIFFERENCE:
                current_invalidity = self.scorer.get_invalidity_degree(self.state[0, :self.n_spins], self.matrix)
                best_invalidity = self.scorer.get_invalidity_degree(self.best_spins[:self.n_spins], self.matrix)
                self.state[idx, :] = (current_invalidity - best_invalidity) / self.scorer._invalidity_normalizer

            elif observable==Observable.VALIDITY_BIT:
                self.state[idx, :] = self.scorer.is_valid(self.state[0, :self.n_spins], self.matrix)


        #############################################################################################
        # 4. Check termination criteria.                                                            #
        #############################################################################################
        if self.current_step == self.max_steps:
            # Maximum number of steps taken --> done.
            # print("Done : maximum number of steps taken")
            done = True

        if not self.reversible_spins:
            if len((self.state[0, :self.n_spins] < 0).nonzero()[0]) == 0:
                # If no more spins to flip --> done.
                # print("Done : no more spins to flip")
                done = True


        return (self.get_observation(), rew, done, None)

    def get_observation(self):
        """
        Returns the state and matrix of observations. Admittedly not sure what matrix of observations does yet. 
        Stacks the state and matrix of observations vertically.
        """
        state = self.state.copy()
        if self.spin_basis == SpinBasis.BINARY:
            # convert {1,-1} --> {0,1}
            state[0,:] = (1-state[0,:])/2

        if self.gg.biased:
            return np.vstack((state, self.matrix_obs, self.bias_obs))
        else:
            return np.vstack((state, self.matrix_obs))

    def get_allowed_action_states(self):
        """
        Gets the allowed action states, either (0, 1) or (1, -1). That is, the states that can be changed (not the result).
        For irreversible ones it's only 0 or 1. Note that for binary basis (0, 1), 0 is considered the default state and -1 is considered
        the default state for signed basis. 
        """
        if self.reversible_spins:
            # If MDP is reversible, both actions are allowed.
            if self.spin_basis == SpinBasis.BINARY:
                return (0,1)
            elif self.spin_basis == SpinBasis.SIGNED:
                return (-1,1)
        else:
            # If MDP is irreversible, only return the state of spins that haven't been flipped.
            if self.spin_basis==SpinBasis.BINARY:
                return 0
            if self.spin_basis==SpinBasis.SIGNED:
                return -1

    def _format_spins_to_signed(self, spins):
        """
        Formats the spins to signed basis, i.e. -1 out, 1 in the candidate solution state
        """
        if self.spin_basis == SpinBasis.BINARY:
            if not np.isin(spins, [0, 1]).all():
                raise Exception("SpinSystem is configured for binary spins ([0,1]).")
            # Convert to signed spins for calculation.
            spins = 2 * spins - 1
        elif self.spin_basis == SpinBasis.SIGNED:
            if not np.isin(spins, [-1, 1]).all():
                raise Exception("SpinSystem is configured for signed spins ([-1,1]).")
        return spins
