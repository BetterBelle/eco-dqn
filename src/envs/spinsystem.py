from abc import ABC, abstractmethod
from collections import namedtuple
from operator import matmul

import numpy as np
import torch.multiprocessing as mp
from numba import jit, float64, int64

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

        if graph_generator.biased:
            return SpinSystemBiased(graph_generator,max_steps,
                                    observables,reward_signal,extra_action,optimisation_target,spin_basis,
                                    norm_rewards,memory_length,horizon_length,stag_punishment,basin_reward,
                                    reversible_spins,
                                    init_snap,seed)
        else:
            return SpinSystemUnbiased(graph_generator,max_steps,
                                      observables,reward_signal,extra_action,optimisation_target,spin_basis,
                                      norm_rewards,memory_length,horizon_length,stag_punishment,basin_reward,
                                      reversible_spins,
                                      init_snap,seed)

class SpinSystemBase(ABC):
    '''
    SpinSystemBase implements the functionality of a SpinSystem that is common to both
    biased and unbiased systems.  Methods that require significant enough changes between
    these two case to not readily be served by an 'if' statement are left abstract, to be
    implemented by a specialised subclass.
    '''

    # Note these are defined at the class level of SpinSystem to ensure that SpinSystem
    # can be pickled.
    class action_space():
        def __init__(self, n_actions):
            self.n = n_actions
            self.actions = np.arange(self.n)

        def sample(self, n=1):
            return np.random.choice(self.actions, n)

    class observation_space():
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

        if self.reward_signal == RewardSignal.SINGLE:
            self.init_score = self.score

        self.best_score = self.score
        self.best_spins = self.state[0,:]

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

        ### Reset the state rewards and greedy actions available, definitely a better name for this function
        self.state = self._reset_state(spins)
        self.score = self.scorer.get_score(self.state[0, :self.n_spins], self.matrix)

        if self.reward_signal == RewardSignal.SINGLE:
            self.init_score = self.score

        ### When resetting, the best score and best observed are just the current score
        self.best_score = self.score
        self.best_obs_score = self.score

        ### TODO: SET THE NORMALIZED SCORE AS WELL SO WE DON'T HAVE TO RECOMPUTE IT

        ### Best solution is the actual solution of the current state, not the score
        self.best_solution = self.scorer.get_solution(self.state[0, :self.n_spins], self.matrix)

        ### Best spins is of course state[0, :self.n_spins] because that holds all the spin booleans similar to score
        self.best_spins = self.state[0, :self.n_spins].copy()
        self.best_obs_spins = self.state[0, :self.n_spins].copy()

        ### Some memory setting that I don't need to worry about because it's just storing scores
        if self.memory_length is not None:
            self.score_memory = np.array([self.best_score] * self.memory_length)
            self.spins_memory = np.array([self.best_spins] * self.memory_length)
            self.idx_memory = 1

        ### Reset graph observables, not really sure what matrix_obs or self.matrix are at the moment
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
        Note that for unreversible spins (i.e. S2V) all spins are set to 1 and can be flipped to -1 but for reversibles
        the spins get set to a random value either 1 or -1. If a spin list is passed in, we format them to be signed.

        Because of matrix multiplication being useful for computing max cut here, they format the spins to floats for parallelisation. 
        """
        state = np.zeros((self.observation_space.shape[1], self.n_actions))

        if spins is None:
            if self.reversible_spins:
                # For reversible spins, initialise randomly to {+1,-1}.
                state[0, :self.n_spins] = 2 * np.random.randint(2, size=self.n_spins) - 1
            else:
                # For irreversible spins, initialise all to +1 (i.e. allowed to be flipped).
                state[0, :self.n_spins] = 1
        else:
            state[0, :] = self._format_spins_to_signed(spins)

        state = state.astype('float')

        # If any observables other than "immediate energy available" require setting to values other than
        # 0 at this stage, we should use a 'for k,v in enumerate(self.observables)' loop.
        for idx, obs in self.observables:
            if obs==Observable.IMMEDIATE_QUALITY_CHANGE:
                ### Normalize the immediately available rewards by the maximum reward to get values between 1 and 0
                state[idx, :self.n_spins] = self.scorer.get_solution_quality_mask(state[0, :self.n_spins], self.matrix) / self.scorer._max_local_reward

            elif obs==Observable.NUMBER_OF_GREEDY_ACTIONS_AVAILABLE:
                immediate_rewards_available = self.scorer.get_score_mask(state[0, :self.n_spins], self.matrix)
                
                ### Every value number of greedy actions available is count of every immediate reward normalized by num vertices
                ### NOTE: 1 - resulting value gives the inverse of rewards <= 0, so the positive rewards
                state[idx, :self.n_spins] = 1 - np.sum(immediate_rewards_available <= 0) / self.n_spins

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

    ### NOTE: NOT CALLED ANYWHERE?
    def calculate_best_energy(self):
        if self.n_spins <= 10:
            # Generally, for small systems the time taken to start multiple processes is not worth it.
            res = self.calculate_best_brute()

        else:
            # Start up processing pool
            n_cpu = int(mp.cpu_count()) / 2

            pool = mp.Pool(mp.cpu_count())

            # Split up state trials across the number of cpus
            iMax = 2 ** (self.n_spins)
            args = np.round(np.linspace(0, np.ceil(iMax / n_cpu) * n_cpu, n_cpu + 1))
            arg_pairs = [list(args) for args in zip(args, args[1:])]

            # Try all the states.
            #             res = pool.starmap(self._calc_over_range, arg_pairs)
            try:
                res = pool.starmap(self._calc_over_range, arg_pairs)
                # Return the best solution,
                idx_best = np.argmin([e for e, s in res])
                res = res[idx_best]
            except Exception as e:
                # Falling back to single-thread implementation.
                # res = self.calculate_best_brute()
                res = self._calc_over_range(0, 2 ** (self.n_spins))
            finally:
                # No matter what happens, make sure we tidy up after outselves.
                pool.close()

            if self.spin_basis == SpinBasis.BINARY:
                # convert {1,-1} --> {0,1}
                best_score, best_spins = res
                best_spins = (1 - best_spins) / 2
                res = best_score, best_spins

            if self.optimisation_target == OptimisationTarget.CUT:
                best_energy, best_spins = res
                best_cut = self.calculate_cut(best_spins)
                res = best_cut, best_spins
            elif self.optimisation_target == OptimisationTarget.ENERGY:
                pass
            else:
                raise NotImplementedError()

        return res

    def seed(self, seed):
        return self.seed

    def set_seed(self, seed):
        self.seed = seed
        np.random.seed(seed)

    def step(self, action):
        done = False

        rew = 0 # Default reward to zero.
        randomised_spins = False
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
                delta_score = new_score - self.score
                self.score = new_score
        else:
            # Calculate score change, which is just the index of the action for the current spins
            delta_score = self.scorer.get_score_mask(self.state[0, :], self.matrix)[action]
            # Now change the state
            new_state[0,action] = -self.state[0,action]
            self.score += delta_score

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
        immediate_quality_change = self.scorer.get_solution_quality_mask(self.state[0, :self.n_spins], self.matrix)
        immediate_score_changes = self.scorer.get_score_mask(self.state[0, :self.n_spins], self.matrix)

        if self.score > self.best_obs_score:
            if self.reward_signal == RewardSignal.BLS:
                rew = self.score - self.best_obs_score
                if self.norm_rewards:
                    rew /= self.n_spins

            ### Don't know what this does as it isn't described in the paper, will ignore for now
            # elif self.reward_signal == RewardSignal.CUSTOM_BLS:
            #     rew = self.score - self.best_obs_score
            #     rew = rew / (rew + 0.1)
            #     if self.norm_rewards:
            #         rew /= self.n_spins
            ###

        ### No idea what this is for, ignore it for now
        # if self.reward_signal == RewardSignal.DENSE:
        #     rew = delta_score
        # elif self.reward_signal == RewardSignal.SINGLE and done:
        #     rew = self.score - self.init_score
        ###

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
            ### TODO: add the normalized version of the score in here as well, so we don't have to recompute it every time
            self.best_score = self.score
            self.best_spins = self.state[0, :self.n_spins].copy()
            self.best_solution = self.scorer.get_solution(self.best_spins, self.matrix)

        if self.memory_length is not None:
            # For case of finite memory length.
            self.score_memory[self.idx_memory] = self.score
            self.spins_memory[self.idx_memory] = self.state[0, :self.n_spins]
            self.idx_memory = (self.idx_memory + 1) % self.memory_length
            self.best_obs_score = self.score_memory.max()
            self.best_obs_spins = self.spins_memory[self.score_memory.argmax()].copy()
        else:
            ### TODO: add the normalized version of the score in here as well, so we don't have to recompute it every time
            self.best_obs_score = self.best_score
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
                self.state[idx, :self.n_spins] = immediate_quality_change / self.scorer._max_local_reward

            elif observable==Observable.TIME_SINCE_FLIP:
                self.state[idx, :] += (1. / self.max_steps)
                if randomised_spins:
                    self.state[idx, :] = self.state[idx, :] * (random_actions > 0)
                else:
                    self.state[idx, action] = 0

            elif observable==Observable.IMMEDIATE_VALIDITY_DIFFERENCE:
                self.state[idx, :] = self.scorer.get_invalidity_degree_mask(self.state[0, :self.n_spins], self.matrix) / self.scorer._invalidity_normalizer

            ### Global observables ###
            elif observable==Observable.EPISODE_TIME:
                self.state[idx, :] += (1. / self.max_steps)

            elif observable==Observable.TERMINATION_IMMANENCY:
                # Update 'Immanency of episode termination'
                self.state[idx, :] = max(0, ((self.current_step - self.max_steps) / self.horizon_length) + 1)

            elif observable==Observable.NUMBER_OF_GREEDY_ACTIONS_AVAILABLE:
                self.state[idx, :] = 1 - np.sum(immediate_score_changes <= 0) / self.n_spins

            elif observable==Observable.DISTANCE_FROM_BEST_SCORE:
                self.state[idx, :] = np.abs(self.score - self.best_obs_score) / self.scorer._max_local_reward

            elif observable==Observable.DISTANCE_FROM_BEST_STATE:
                self.state[idx, :] = np.count_nonzero(self.best_obs_spins[:self.n_spins] - self.state[0, :self.n_spins])

            elif observable==Observable.GLOBAL_VALIDITY_DIFFERENCE:
                self.state[idx, :] = self.get_edges_uncovered(self.state[0, :self.n_spins], self.matrix) - self.get_edges_uncovered(self.best_obs_spins, self.matrix)

            elif observable==Observable.VALIDITY_BIT:
                # if number of edges uncovered is 0, then valid (True)
                self.state[idx, :] = self.get_edges_uncovered(self.state[0, :self.n_spins], self.matrix) == 0


        #############################################################################################
        # 4. Check termination criteria.                                                            #
        #############################################################################################
        if self.current_step == self.max_steps:
            # Maximum number of steps taken --> done.
            # print("Done : maximum number of steps taken")
            done = True

        if not self.reversible_spins:
            if len((self.state[0, :self.n_spins] > 0).nonzero()[0]) == 0:
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

    def get_immediate_rewards_available(self, spins=None):
        """
        Returns the list of immediate rewards available given the spins. For example, for the maximum cut this returns the list of 
        rewards (i.e. new cut values) for each vertex if flipped. Note that positive values are good, so for minimization problems these
        functions should return positive values for improved (smaller) solutions.
        """
        if spins is None:
            spins = self._get_spins()

        if self.optimisation_target==OptimisationTarget.ENERGY:
            immediate_reward_function = lambda *args: -1*self._get_immeditate_energies_avaialable_jit(*args)
        elif self.optimisation_target==OptimisationTarget.CUT:
            immediate_reward_function = self._get_immeditate_cuts_avaialable_jit
        elif self.optimisation_target == OptimisationTarget.MIN_COVER:
            immediate_reward_function = SpinSystemUnbiased._get_immediate_vertex_covers_available
        else:
            raise NotImplementedError("Optimisation target {} not recognised.".format(self.optimisation_target))

        spins = spins.astype('float64')
        matrix = self.matrix.astype('float64')

        if self.gg.biased:
            bias = self.bias.astype('float64')
            return immediate_reward_function(spins,matrix,bias)
        else:
            return immediate_reward_function(spins,matrix)

    def get_allowed_action_states(self):
        """
        Gets the allowed actions, either (0, 1) or (1, -1).
        For irreversible ones it's only 0 or 1. Note that for binary basis (0, 1), 1 is considered the default state and -1 is considered
        the default state for signed basis. 
        """
        if self.reversible_spins:
            # If MDP is reversible, both actions are allowed.
            if self.spin_basis == SpinBasis.BINARY:
                return (0,1)
            elif self.spin_basis == SpinBasis.SIGNED:
                return (1,-1)
        else:
            # If MDP is irreversible, only return the state of spins that haven't been flipped.
            if self.spin_basis==SpinBasis.BINARY:
                return 0
            if self.spin_basis==SpinBasis.SIGNED:
                return 1

    def calculate_score(self, spins=None):
        """
        Calculates the score given the current spin states depending on the optimization target
        """
        if self.optimisation_target==OptimisationTarget.CUT:
            score = self.calculate_cut(spins)
        elif self.optimisation_target==OptimisationTarget.ENERGY:
            score = -1.*self.calculate_energy(spins)
        elif self.optimisation_target==OptimisationTarget.MIN_COVER:
            score = self.calculate_mvc(spins)
        else:
            raise NotImplementedError
        return score

    def _calculate_score_change(self, new_spins, matrix, action):
        if self.optimisation_target==OptimisationTarget.CUT:
            delta_score = self._calculate_cut_change(new_spins, matrix, action)
        elif self.optimisation_target == OptimisationTarget.ENERGY:
            delta_score = -1. * self._calculate_energy_change(new_spins, matrix, action)
        elif self.optimisation_target == OptimisationTarget.MIN_COVER:
            delta_score = self._calculate_mvc_score_change(new_spins, matrix, action)
        else:
            raise NotImplementedError
        return delta_score

    def _format_spins_to_signed(self, spins):
        if self.spin_basis == SpinBasis.BINARY:
            if not np.isin(spins, [0, 1]).all():
                raise Exception("SpinSystem is configured for binary spins ([0,1]).")
            # Convert to signed spins for calculation.
            spins = 2 * spins - 1
        elif self.spin_basis == SpinBasis.SIGNED:
            if not np.isin(spins, [-1, 1]).all():
                raise Exception("SpinSystem is configured for signed spins ([-1,1]).")
        return spins
    
    def get_edges_uncovered(self, spins, matrix):
        """
        Number of edges not covered by spins
        """
        return np.sum((matrix != 0) * (spins == -1) * np.array([spins == -1], dtype=np.float64).T) / 2

    @abstractmethod
    def calculate_energy(self, spins=None):
        raise NotImplementedError

    @abstractmethod
    def calculate_cut(self, spins=None):
        raise NotImplementedError
    
    @abstractmethod
    def calculate_mvc(self, spins=None):
        raise NotImplementedError

    @abstractmethod
    def get_best_cut(self):
        raise NotImplementedError

    @abstractmethod
    def _calc_over_range(self, i0, iMax):
        raise NotImplementedError

    @abstractmethod
    def _calculate_energy_change(self, new_spins, matrix, action):
        raise NotImplementedError

    @abstractmethod
    def _calculate_cut_change(self, new_spins, matrix, action):
        raise NotImplementedError
    
    @abstractmethod
    def _calculate_mvc_score_change(self, new_spins, matrix, action):
        raise NotImplementedError
    
    @abstractmethod
    def get_newly_uncovered_edges(self, spins, matrix):
        raise NotImplementedError

##########
# Classes for implementing the calculation methods with/without biases.
##########
class SpinSystemUnbiased(SpinSystemBase):

    def calculate_energy(self, spins=None):
        if spins is None:
            spins = self._get_spins()
        else:
            spins = self._format_spins_to_signed(spins)

        spins = spins.astype('float64')
        matrix = self.matrix.astype('float64')

        return self._calculate_energy_jit(spins, matrix)

    def calculate_cut(self, spins=None):
        """
        Calculate the cut value given the spins chosen
        """
        if spins is None:
            spins = self._get_spins()
        else:
            spins = self._format_spins_to_signed(spins)

        return (1/4) * np.sum( np.multiply( self.matrix, 1 - np.outer(spins, spins) ) )

    def get_best_cut(self):
        if self.optimisation_target==OptimisationTarget.CUT or self.optimisation_target==OptimisationTarget.MIN_CUT:
            return self.best_score
        else:
            raise NotImplementedError("Can't return best cut when optimisation target is set to energy.")
        
    def calculate_mvc(self, spins = None):
        """
        Calculate the size of the current cover regardless of validity. 
        """
        if spins is None:
            spins = self._get_spins()
        
        return np.sum(spins == 1)

    def _calc_over_range(self, i0, iMax):
        list_spins = [2 * np.array([int(x) for x in list_string]) - 1
                      for list_string in
                      [list(np.binary_repr(i, width=self.n_spins))
                       for i in range(int(i0), int(iMax))]]
        matrix = self.matrix.astype('float64')
        return self.__calc_over_range_jit(list_spins, matrix)

    @staticmethod
    @jit(float64(float64[:],float64[:,:],int64), nopython=True)
    def _calculate_energy_change(new_spins, matrix, action):
        return -2 * new_spins[action] * matmul(new_spins.T, matrix[:, action])

    @staticmethod
    @jit(float64(float64[:],float64[:,:],int64), nopython=True)
    def _calculate_cut_change(new_spins, matrix, action):
        return -1 * new_spins[action] * matmul(new_spins.T, matrix[:, action])
    
    @staticmethod
    def _calculate_mvc_score_change(new_spins, matrix, action):
        """
        Given array of new_spins and adjacency matrix, find the score change given the spin "action" was changed
        This is just the -1 * immediate_vertex_covers_available(old_spins), where old spins is just flipping the action
        """
        old_spins = new_spins
        old_spins[action] *= -1

        return -1 * SpinSystemUnbiased._get_immediate_vertex_covers_available(old_spins, matrix)[action]

    @staticmethod
    @jit(float64(float64[:],float64[:,:]), nopython=True)
    def _calculate_energy_jit(spins, matrix):
        return - matmul(spins.T, matmul(matrix, spins)) / 2

    @staticmethod
    @jit(parallel=True)
    def __calc_over_range_jit(list_spins, matrix):
        energy = 1e50
        best_spins = None

        for spins in list_spins:
            spins = spins.astype('float64')
            # This is self._calculate_energy_jit without calling to the class or self so jit can do its thing.
            current_energy = - matmul(spins.T, matmul(matrix, spins)) / 2
            if current_energy < energy:
                energy = current_energy
                best_spins = spins
        return energy, best_spins

    @staticmethod
    @jit(float64[:](float64[:],float64[:,:]), nopython=True)
    def _get_immeditate_energies_avaialable_jit(spins, matrix):
        return 2 * spins * matmul(matrix, spins)

    @staticmethod
    @jit(float64[:](float64[:],float64[:,:]), nopython=True)
    def _get_immeditate_cuts_avaialable_jit(spins, matrix):
        return spins * matmul(matrix, spins)
    
    @staticmethod
    def _get_immediate_vertex_covers_available(spins, matrix):
        """
        ### Explanation: matrix multiplication of adj and spins (vertices) gives number of edges incident on that node
        ### spins == -1 gives vertices that are not in the solution as boolean mask
        ### Multiplying by matrix gets rid of all edges covered by nodes in solution (0s) without double counting, column wise
        ### That result is essentially a graph removing the edges covered by the solution (but only counting once)
        ### This new matrix multiplied by spins therefore gives the number of edges not covered by another vertex in the solution incident on
            each vertex
        ### Multiplying this by the state of spins (1s in solution and -1s not in solution) gives the change in number of edges covered on flip
        ### Sum of an adjacency matrix divided by two is the number of edges in that matrix
        ### We want the number of edges that aren't covered by our solution (because our list counts only those edges anyway)
        ### We can do this by multiplying by the bool mask of spins in solution and it's transpose
        ### If we then take that value and subtract from it the change in covered edges for every vertex, we get a value that represents 
        ### the number of vertices not covered by the solution on each vertex flip
        ### Any value that is a 0 means by flipping that vertex you get a valid cover
        ### We want the immediate reward to be the size of the cover on flip, but have it be -1 when invalid

        # We want: If an invalid solution is created, the reward for that flip should be the difference in degree of invalidity
        # If a valid solution is created, it should be the difference in size of the set
        """
        newly_covered_on_flip = SpinSystemUnbiased.get_newly_uncovered_edges(spins, matrix)
        uncovered_edges = np.sum((matrix) * (spins == -1) * np.array([spins == -1], dtype=np.float64).T) / 2
        total_uncovered_on_flip = uncovered_edges - newly_covered_on_flip
        solution_set_size = sum(spins == 1)


        # First get a boolean mask for validity
        # To verify this, check if the number of uncovered edges minus the number of edges covered on flip (it's negative if it loses edges) is 0
        validity_mask = total_uncovered_on_flip == 0

        # Subtracting spins from solution set size gives you the new set size on flip
        # We then take the total number of nodes and subtract from that the new set size to give you the size of the set not in the solution,
        # this ensures that larger solution sets give smaller rewards (i.e. a set size of 3 with 5 nodes vs a set size of 2 with 5 nodes will now
        # give rewards of 2 and 3 (higher reward for smaller set) instead of 3 and 2)
        # Next, we multiply by the validity mask to keep only the valid solutions
        new_set_size_score = validity_mask * (len(spins) - (solution_set_size - spins))
        
        # Next we want the new number of uncovered edges on flip (i.e. the new degree of invalidity) for each flip
        # This is simply the uncovered edges - newly covered on flip
        new_uncovered_edges =  uncovered_edges - newly_covered_on_flip

        # Score is defined as the set score - new_uncovered edges
        score_on_flip = new_set_size_score - new_uncovered_edges

        # Now, these rewards are relative to the current state, so calculate the score for the current state
        current_validity = (uncovered_edges == 0)
        current_score = current_validity * (len(spins) - solution_set_size) - uncovered_edges

        # Therefore immediate rewards are going to be the difference between the current score and the different scores on each flip
        immediate_rewards_available = score_on_flip - current_score

        return immediate_rewards_available
    
    @staticmethod
    @jit(nopython=True)
    def get_newly_uncovered_edges(spins, matrix):
        return spins * matmul(matrix * (spins == -1), spins)

class SpinSystemBiased(SpinSystemBase):

    def calculate_energy(self, spins=None):
        if type(spins) == type(None):
            spins = self._get_spins()

        spins = spins.astype('float64')
        matrix = self.matrix.astype('float64')
        bias = self.bias.astype('float64')

        return self._calculate_energy_jit(spins, matrix, bias)

    def calculate_cut(self, spins=None):
        raise NotImplementedError("MaxCut not defined/implemented for biased SpinSystems.")

    def get_best_cut(self):
        raise NotImplementedError("MaxCut not defined/implemented for biased SpinSystems.")

    def _calc_over_range(self, i0, iMax):
        list_spins = [2 * np.array([int(x) for x in list_string]) - 1
                      for list_string in
                      [list(np.binary_repr(i, width=self.n_spins))
                       for i in range(int(i0), int(iMax))]]
        matrix = self.matrix.astype('float64')
        bias = self.bias.astype('float64')
        return self.__calc_over_range_jit(list_spins, matrix, bias)

    @staticmethod
    @jit(nopython=True)
    def _calculate_energy_change(new_spins, matrix, bias, action):
        return 2 * new_spins[action] * (matmul(new_spins.T, matrix[:, action]) + bias[action])

    @staticmethod
    @jit(nopython=True)
    def _calculate_cut_change(new_spins, matrix, bias, action):
        raise NotImplementedError("MaxCut not defined/implemented for biased SpinSystems.")

    @staticmethod
    @jit(nopython=True)
    def _calculate_energy_jit(spins, matrix, bias):
        return matmul(spins.T, matmul(matrix, spins))/2 + matmul(spins.T, bias)

    @staticmethod
    @jit(parallel=True)
    def __calc_over_range_jit(list_spins, matrix, bias):
        energy = 1e50
        best_spins = None

        for spins in list_spins:
            spins = spins.astype('float64')
            # This is self._calculate_energy_jit without calling to the class or self so jit can do its thing.
            current_energy = -( matmul(spins.T, matmul(matrix, spins))/2 + matmul(spins.T, bias))
            if current_energy < energy:
                energy = current_energy
                best_spins = spins
        return energy, best_spins

    @staticmethod
    @jit(nopython=True)
    def _get_immeditate_energies_avaialable_jit(spins, matrix, bias):
        return - (2 * spins * (matmul(matrix, spins) + bias))

    @staticmethod
    @jit(nopython=True)
    def _get_immeditate_cuts_avaialable_jit(spins, matrix, bias):
        raise NotImplementedError("MaxCut not defined/implemented for biased SpinSystems.")