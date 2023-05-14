from abc import ABC, abstractmethod
from numpy.typing import ArrayLike, NDArray
from src.envs.utils import OptimisationTarget
from numba import jit, float64

import operator as op
import src.envs.utils as env_utils
import numpy as np
import numpy.typing as npt


class ScoreSolver(ABC):
    """
    Class representing the different functions required for determining scores of solutions, including validity and invalidity.
    """
    def __init__(self, problem_type : env_utils.OptimisationTarget, is_biased_graph : bool):
        self._problem_type = problem_type
        self._is_biased_graph = is_biased_graph
        self._max_local_reward = 1
        self._solution_quality_normalizer = 1
        self._invalidity_normalizer = 1

    @abstractmethod
    def set_quality_normalizer(self, spins : npt.ArrayLike, matrix : npt.ArrayLike) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_max_local_reward(self, spins : npt.ArrayLike, matrix : npt.ArrayLike) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def set_invalidity_normalizer(self, spins : npt.ArrayLike, matrix : npt.ArrayLike) -> None:
        raise NotImplementedError

    def get_validity_mask(self, spins : npt.ArrayLike, matrix : npt.ArrayLike) -> npt.NDArray:
        """
        Get the validity bit for what happens on each vertex flip in spins.
        This must therefore be True if that flip results in a valid solution or False if it results
        in an invalid solution.
        Valid solutions have an invalidity degree of 0, therefore we just 
        """
        return np.all(self.get_invalidity_degree_mask(spins, matrix) == 0)


    def is_valid(self, spins : npt.ArrayLike, matrix : npt.ArrayLike) -> bool:
        """
        Determines whether the passed in spins are a valid solution.
        For any problem, this is True if the invalidity degree of the passed in spins equals 0 and False otherwise.
        """
        return self.get_invalidity_degree(spins, matrix) == 0
    
    @abstractmethod
    def get_solution_quality_mask(self, spins : npt.ArrayLike, matrix : npt.ArrayLike) -> npt.NDArray:
        """
        Gets the solution quality (how good the solution is, irrespective of validity) for every single vertex if flipped.
        This is computed with respect to the current spin's quality.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_solution_quality(self, spins : npt.ArrayLike, matrix : npt.ArrayLike) -> float:
        """
        Gets the solution quality of the current spins, irrespective of validity
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_invalidity_degree_mask(self, spins : npt.ArrayLike, matrix : npt.ArrayLike) -> npt.NDArray:
        """
        Gets the degree of invalidity for every spin if it was flipped. 
        This is all done with respect to the passed in spins, so it is
        a difference in invalidity with respect to the current spins.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_invalidity_degree(self, spins : npt.ArrayLike, matrix : npt.ArrayLike) -> float:
        """
        Gets the degree of invalidity of the passed in spins. Note that this is 0 for a valid solution.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_score_mask(self, spins : npt.ArrayLike, matrix : npt.ArrayLike) -> npt.NDArray:
        """
        Get the score for every spin flip. This is the difference in score from the current spins.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_score(self, spins : npt.ArrayLike, matrix : npt.ArrayLike) -> float:
        """
        Get the score of the passed in spins. 
        """
        raise NotImplementedError
    


class MinimumVertexCoverUnbiasedScorer(ScoreSolver):
    def __init__(self, problem_type: OptimisationTarget, is_biased_graph: bool):
        super().__init__(problem_type, is_biased_graph)
        raise NotImplementedError("This optimizer is not implemented : %s", problem_type)


    def get_solution_quality_mask(self, spins: ArrayLike, matrix: ArrayLike) -> NDArray:
        """
        For minimum vertex cover, the solution quality for each spin flip is the current solution quality plus the spin state.
        Because spins states are 1 for in the solution and -1 for not in, we want higher scores for smaller sets, therefore flipping the 
        state of a vertex already in the solution increases the quality by 1, we want to add it and vice versa for those not in the set.
        """
        return self.get_solution_quality(spins, matrix) + spins
    

    def get_solution_quality(self, spins: ArrayLike, matrix: ArrayLike) -> float:
        """
        For minimum vertex cover, solution quality is just the number of vertices in the graph minus the size of the solution set.
        Reason we do this is to grant higher scores to smaller set sizes. 
        """
        return len(spins) - np.sum(spins == 1)
    

    def get_invalidity_degree_mask(self, spins: ArrayLike, matrix: ArrayLike) -> NDArray:
        return 
    
    def get_invalidity_degree(self, spins: ArrayLike, matrix: ArrayLike) -> float:
        return
        


class MinimumVertexCoverBiasedScorer(ScoreSolver):
    def __init__(self, problem_type, is_biased_graph):
        super().__init__(problem_type, is_biased_graph)
        raise NotImplementedError("This optimizer is not implemented : %s", problem_type)



class MaximumCutUnbiasedScorer(ScoreSolver):
    def __init__(self, problem_type, is_biased_graph):
        super().__init__(problem_type, is_biased_graph)

    ### Following methods are private static jit methods for speedup
    @staticmethod
    @jit(float64[:](float64[:],float64[:,:]), nopython=True)
    def _solution_quality_mask_calculator(spins : ArrayLike, matrix: ArrayLike) -> NDArray:
        return spins * op.matmul(matrix, spins)

    def set_invalidity_normalizer(self, spins: ArrayLike, matrix: ArrayLike) -> None:
        """
        Maximum cut does not need this so we just set it to 1
        """
        self._invalidity_normalizer = 1

    def set_quality_normalizer(self, spins: ArrayLike, matrix: ArrayLike) -> None:
        """
        Quality normalizer is the number of vertices for the maximum cut. 
        """
        self._solution_quality_normalizer = len(spins)

    def set_max_local_reward(self, spins: ArrayLike, matrix: ArrayLike) -> None:
        """
        For the maximum cut, the maximum local reward is the maximum immediate reward from the empty set (i.e. the highest weight
        on any node). This is just calling the solution quality mask with an empty (all -1) solution set. Can also be the score mask,
        but that just results in more function calls as the score is just the quality due to having no 
        """
        quality_mask = self.get_solution_quality_mask(np.array([-1] * len(spins), dtype=np.float64), matrix)
        without_zeros = quality_mask[np.nonzero(quality_mask)]
        self._max_local_reward = np.max(without_zeros)

    def get_solution_quality_mask(self, spins: ArrayLike, matrix: ArrayLike) -> NDArray:
        """
        For the maximum cut, this will be the change in cut value
        for each spin flip compared to the passed spins.
        """
        return MaximumCutUnbiasedScorer._solution_quality_mask_calculator(spins, matrix)
    
    def get_solution_quality(self, spins: ArrayLike, matrix: ArrayLike) -> float:
        """
        Solution quality here is just the cut value
        """
        return (1/4) * np.sum(np.multiply(matrix, 1 - np.outer(spins, spins)))

    def get_invalidity_degree(self, spins: ArrayLike, matrix: ArrayLike) -> float:
        """
        There is no such thing as an invalid solution for the max cut.
        Therefore we just set the invalidity degree to 0.
        """
        return 0
    
    def get_invalidity_degree_mask(self, spins: ArrayLike, matrix: ArrayLike) -> NDArray:
        """
        All max cut solutions are valid. Therefore the invalidity degree is all 0s.
        """
        return [0 for _ in range(len(spins))]
    
    def get_score_mask(self, spins: ArrayLike, matrix: ArrayLike) -> NDArray:
        """
        Score mask for the case of maximum cut is the same as the solution quality mask.
        """
        return self.get_solution_quality_mask(spins, matrix)
    
    def get_score(self, spins: ArrayLike, matrix: ArrayLike) -> float:
        """
        Score is the same as the quality for max cut.
        """
        return self.get_solution_quality(spins, matrix)
    


class MaximumCutBiasedScorer(ScoreSolver):
    def __init__(self, problem_type, is_biased_graph):
        super().__init__(problem_type, is_biased_graph)
        raise NotImplementedError("This optimizer is not implemented : %s", problem_type)



class EnergyUnbiasedScorer(ScoreSolver):
    def __init__(self, problem_type, is_biased_graph):
        super().__init__(problem_type, is_biased_graph)
        raise NotImplementedError("This optimizer is not implemented : %s", problem_type)



class EnergyBiasedScorer(ScoreSolver):
    def __init__(self, problem_type, is_biased_graph):
        super().__init__(problem_type, is_biased_graph)
        raise NotImplementedError("This optimizer is not implemented : %s", problem_type)



class MinimumCutUnbiasedSolver(ScoreSolver):
    def __init__(self, problem_type: OptimisationTarget, is_biased_graph: bool):
        super().__init__(problem_type, is_biased_graph)

    ### Following methods are private static jit methods for speedup
    @staticmethod
    @jit(float64[:](float64[:],float64[:,:]), nopython=True)
    def _solution_quality_mask_calculator(spins : ArrayLike, matrix: ArrayLike) -> NDArray:
        """
        Because this is the minimum cut, we want the 
        """
        return - (spins * op.matmul(matrix, spins))

    def set_invalidity_normalizer(self, spins: ArrayLike, matrix: ArrayLike) -> None:
        """
        Maximum cut does not need this so we just set it to 1
        """
        self._invalidity_normalizer = 1

    def set_quality_normalizer(self, spins: ArrayLike, matrix: ArrayLike) -> None:
        """
        Quality normalizer is the number of vertices for the maximum cut. 
        """
        self._solution_quality_normalizer = len(spins)

    def set_max_local_reward(self, spins: ArrayLike, matrix: ArrayLike) -> None:
        """
        For the minimum cut, the maximum local reward is the maximum immediate reward from the empty set (i.e. the highest weight
        on any node). This is just calling the solution quality mask with an empty (all -1) solution set. Can also be the score mask,
        but that just results in more function calls as the score is just the quality due to having no 
        """
        quality_mask = self.get_solution_quality_mask(np.array([-1] * len(spins), dtype=np.float64), matrix)
        without_zeros = quality_mask[np.nonzero(quality_mask)]
        self._max_local_reward = np.max(without_zeros)

    def get_solution_quality_mask(self, spins: ArrayLike, matrix: ArrayLike) -> NDArray:
        """
        For the maximum cut, this will be the change in cut value
        for each spin flip compared to the passed spins.
        """
        return MinimumCutUnbiasedSolver._solution_quality_mask_calculator(spins, matrix)
    
    def get_solution_quality(self, spins: ArrayLike, matrix: ArrayLike) -> float:
        """
        Solution quality here is just the cut value, but because it's the minimum cut and everything else is set to maximize,
        we'll just flip the sign so it gives greater rewards to lower cut values.
        """
        return - (1/4) * np.sum(np.multiply(matrix, 1 - np.outer(spins, spins)))

    def get_invalidity_degree(self, spins: ArrayLike, matrix: ArrayLike) -> float:
        """
        There is no such thing as an invalid solution for the max cut.
        Therefore we just set the invalidity degree to 0.
        """
        return 0
    
    def get_invalidity_degree_mask(self, spins: ArrayLike, matrix: ArrayLike) -> NDArray:
        """
        All max cut solutions are valid. Therefore the invalidity degree is all 0s.
        """
        return [0 for _ in range(len(spins))]
    
    def get_score_mask(self, spins: ArrayLike, matrix: ArrayLike) -> NDArray:
        """
        Score mask for the case of minimum cut is the same as the solution quality mask.
        """
        return self.get_solution_quality_mask(spins, matrix)
    
    def get_score(self, spins: ArrayLike, matrix: ArrayLike) -> float:
        """
        Score is the same as the quality for minimum cut.
        """
        return self.get_solution_quality(spins, matrix)


class ScoreSolverFactory():
    @staticmethod
    def get(problem_type : env_utils.OptimisationTarget, is_biased_graph : bool) -> ScoreSolver:
        """
        Creates a ScoreSolver depending on the optimization target and whether the systems are biased (i.e. digraphs)
        """
        if problem_type == env_utils.OptimisationTarget.CUT and not is_biased_graph:
            return MaximumCutUnbiasedScorer(problem_type, is_biased_graph)
        
        elif problem_type == env_utils.OptimisationTarget.CUT and is_biased_graph:
            return MaximumCutUnbiasedScorer(problem_type, is_biased_graph)
        
        elif problem_type == env_utils.OptimisationTarget.MIN_COVER and not is_biased_graph:
            return MinimumVertexCoverUnbiasedScorer(problem_type, is_biased_graph)
        
        elif problem_type == env_utils.OptimisationTarget.MIN_COVER and is_biased_graph:
            return MinimumVertexCoverBiasedScorer(problem_type, is_biased_graph)
        
        elif problem_type == env_utils.OptimisationTarget.ENERGY and not is_biased_graph:
            return EnergyUnbiasedScorer(problem_type, is_biased_graph)
        
        elif problem_type == env_utils.OptimisationTarget.ENERGY and is_biased_graph:
            return EnergyBiasedScorer(problem_type, is_biased_graph)
        
        elif problem_type == env_utils.OptimisationTarget.MIN_CUT and not is_biased_graph:
            return MinimumCutUnbiasedSolver(problem_type, is_biased_graph)
        
        # If make it here, invalid target
        raise NotImplementedError("Invalid optimization target: %s and biased %s", problem_type, is_biased_graph)