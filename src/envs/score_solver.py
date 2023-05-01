from abc import ABC, abstractmethod
from numpy.typing import ArrayLike, NDArray
from src.envs.utils import OptimisationTarget

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


    def get_validity_mask(self, spins : npt.ArrayLike, matrix : npt.ArrayLike) -> npt.NDArray:
        """
        Get the validity bit for what happens on each vertex flip in spins.
        For any problem, this is the invalidity degree mask with True whenever the value is 0.
        """
        return self.get_invalidity_degree_mask(spins, matrix) == 0


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
        Gets the degree of invalidity for every spin if it was flipped. Note a value cannot be greater than 0 (a valid solution).
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_invalidity_degree(self, spins : npt.ArrayLike, matrix : npt.ArrayLike) -> float:
        """
        Gets the degree of invalidity of the passed in spins. Note that this is 0 for a valid solution.
        """
        raise NotImplementedError
    
    
    def get_score_mask(self, spins : npt.ArrayLike, matrix : npt.ArrayLike) -> npt.NDArray:
        """
        Get the score for every spin flip. For any problem this is
        validity * solution quality - degree of invalidity

        Thankfully numpy allows us to do this all elementwise on all our other functions.
        """
        return self.get_validity_mask(spins, matrix) * self.get_solution_quality_mask(spins, matrix) - self.get_invalidity_degree_mask(spins, matrix)
    
    
    def get_score(self, spins : npt.ArrayLike, matrix : npt.ArrayLike) -> float:
        """
        Get the score of the passed in spins. For any problem this is 
        validity * solution quality - degree of invalidity
        """
        return self.is_valid(spins, matrix) * self.get_solution_quality(spins, matrix) - self.get_invalidity_degree(spins, matrix)
    


class MinimumVertexCoverUnbiasedScorer(ScoreSolver):
    def __init__(self, problem_type: OptimisationTarget, is_biased_graph: bool):
        super().__init__(problem_type, is_biased_graph)


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



class MaximumCutUnbiasedScorer(ScoreSolver):
    def __init__(self, problem_type, is_biased_graph):
        super().__init__(problem_type, is_biased_graph)



class MaximumCutBiasedScorer(ScoreSolver):
    def __init__(self, problem_type, is_biased_graph):
        super().__init__(problem_type, is_biased_graph)



class EnergyUnbiasedScorer(ScoreSolver):
    def __init__(self, problem_type, is_biased_graph):
        super().__init__(problem_type, is_biased_graph)



class EnergyBiasedScorer(ScoreSolver):
    def __init__(self, problem_type, is_biased_graph):
        super().__init__(problem_type, is_biased_graph)



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
        
        # If make it here, invalid target
        raise NotImplementedError("Invalid optimization target: %s", problem_type)