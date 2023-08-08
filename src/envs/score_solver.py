from abc import ABC, abstractmethod
from numpy.typing import ArrayLike, NDArray
from src.envs.utils import OptimisationTarget, calculate_cut, calculate_cut_changes

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
        self._lower_bound = 0

    @abstractmethod
    def set_quality_normalizer(self, spins : npt.ArrayLike, matrix : npt.ArrayLike) -> None:
        """
        ABSTRACT 

        This will normalize the quality for the normalized score. Doubles as an "upper bound" for solutions
        """
        raise NotImplementedError

    @abstractmethod
    def set_max_local_reward(self, spins : npt.ArrayLike, matrix : npt.ArrayLike) -> None:
        """
        ABSTRACT 

        This is the normalizer for the local quality changes. 
        """
        raise NotImplementedError
    
    @abstractmethod
    def set_invalidity_normalizer(self, spins : npt.ArrayLike, matrix : npt.ArrayLike) -> None:
        """
        ABSTRACT 

        To normalize the invalidity degree
        """
        raise NotImplementedError

    @abstractmethod
    def set_lower_bound(self, spins : npt.ArrayLike, matrix : npt.ArrayLike) -> None:
        """
        ABSTRACT

        Setting a lower bound to shift measures for scores
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_solution(self, spins : npt.ArrayLike, matrix : npt.ArrayLike) -> float:
        """
        ABSTRACT 
        
        Gets the actual solution score for the problem at hand given the spins and matrix. This is important for problems
        where the score/reward will be different than the actual solution, like for minimization problems for example.
        """
        raise NotImplementedError
    
    @abstractmethod
    def _get_measure(self, spins : npt.ArrayLike, matrix : npt.ArrayLike) -> float:
        """
        ABSTRACT

        Gets the measure of the solution without accounting for validity. Only for use
        in calculating score.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_solution_quality_mask(self, spins : npt.ArrayLike, matrix : npt.ArrayLike) -> npt.NDArray:
        """
        ABSTRACT 
        
        Gets the solution quality (how good the solution is, irrespective of validity) for every single vertex if flipped.
        This is computed with respect to the current spin's quality.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_solution_quality(self, spins : npt.ArrayLike, matrix : npt.ArrayLike) -> float:
        """
        ABSTRACT 
        
        Gets the solution quality of the current spins, irrespective of validity
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_invalidity_degree_mask(self, spins : npt.ArrayLike, matrix : npt.ArrayLike) -> npt.NDArray:
        """
        ABSTRACT 
        
        Gets the degree of invalidity for every spin if it was flipped. 
        This is all done with respect to the passed in spins, so it is
        a difference in invalidity with respect to the current spins.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_invalidity_degree(self, spins : npt.ArrayLike, matrix : npt.ArrayLike) -> float:
        """
        ABSTRACT 
        
        Gets the degree of invalidity of the passed in spins. Note that this is 0 for a valid solution.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_score_mask(self, spins : npt.ArrayLike, matrix : npt.ArrayLike) -> npt.NDArray:
        """
        ABSTRACT 
        
        Get the score for every spin flip. This is the difference in score from the current spins.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_normalized_score_mask(self, spins : npt.ArrayLike, matrix : npt.ArrayLike) -> float:
        """
        ABSTRACT 
        
        Gets the score difference for every vertex, normalized.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_score(self, spins : npt.ArrayLike, matrix : npt.ArrayLike) -> float:
        """
        ABSTRACT

        Get the score of the passed in spins. This is always just is_valid * quality - invalidity.
        Differs between max/min problems
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_normalized_score(self, spins : npt.ArrayLike, matrix : npt.ArrayLike) -> float:
        """
        ABSTRACT

        Get the normalized score of the passed in spins. This is always just is_valid * quality - invalidity.
        Differs between max/min problems
        """
        raise NotImplementedError
    
    def get_validity_mask(self, spins : npt.ArrayLike, matrix : npt.ArrayLike) -> npt.NDArray:
        """
        Get the validity bit for what happens on each vertex flip in spins.
        """
        # Because the invalidity_degree is the current invalidity degree and the invalidity_degree_mask is the change,
        # (negative getting closer to validity), we can just add the current validity degree to the invalidity degree mask,
        # giving us the new degree of invalidity for each flip. Checking which ones are zeros gives us the bitmask for valid solutions.
        new_validity_degree = self.get_invalidity_degree(spins, matrix) + self.get_invalidity_degree_mask(spins, matrix)
        return new_validity_degree == 0

    def is_valid(self, spins : npt.ArrayLike, matrix : npt.ArrayLike) -> bool:
        """
        Determines whether the passed in spins are a valid solution.
        For any problem, this is True if the invalidity degree of the passed in spins equals 0 and False otherwise.
        """
        return self.get_invalidity_degree(spins, matrix) == 0
    


class MaximizationProblem(ScoreSolver):
    """
    Class template for a maximization problem
    """
    def __init__(self, problem_type: OptimisationTarget, is_biased_graph: bool):
        super().__init__(problem_type, is_biased_graph)

    def get_score(self, spins: ArrayLike, matrix: ArrayLike) -> float:
        """
        Score is defined by is_valid * quality - invalidity

        Quality being the measure score.
        """
        return self.is_valid(spins, matrix) * self.get_solution_quality(spins, matrix) - self.get_invalidity_degree(spins, matrix)

    def get_normalized_score(self, spins: ArrayLike, matrix: ArrayLike) -> float:
        """
        Normalizing the score. 
        """
        return self.is_valid(spins, matrix) * self.get_solution_quality(spins, matrix) / self._solution_quality_normalizer - self.get_invalidity_degree(spins, matrix) / self._invalidity_normalizer

    def get_solution_quality(self, spins: ArrayLike, matrix: ArrayLike) -> float:
        """
        Solution quality (or measure score) is always obtained the same for a maximization problem. measure + abs(min(0, LB))
        """
        return self._get_measure(spins, matrix) + abs(min(0, self._lower_bound))


class MinimizationProblem(ScoreSolver):
    """
    Class template for a minimization problem
    """
    def __init__(self, problem_type: OptimisationTarget, is_biased_graph: bool):
        super().__init__(problem_type, is_biased_graph)

    def get_score(self, spins: ArrayLike, matrix: ArrayLike) -> float:
        """
        Score is defined by is_valid * quality - invalidity

        Quality being the measure score.
        """
        return self.is_valid(spins, matrix) * self.get_solution_quality(spins, matrix) - self.get_invalidity_degree(spins, matrix)

    def get_normalized_score(self, spins: ArrayLike, matrix: ArrayLike) -> float:
        """
        Normalizing the score. 
        """
        return self.is_valid(spins, matrix) * self.get_solution_quality(spins, matrix) / self._solution_quality_normalizer - self.get_invalidity_degree(spins, matrix) / self._invalidity_normalizer

    def get_solution_quality(self, spins: ArrayLike, matrix: ArrayLike) -> float:
        """
        Solution quality (or measure score) is always obtained the same for a minimization problem. - measure + max(0, UB))
        """
        return max(0, self._solution_quality_normalizer) - self._get_measure(spins, matrix)



class MinimumVertexCoverUnbiasedScorer(MinimizationProblem):
    def __init__(self, problem_type: OptimisationTarget, is_biased_graph: bool):
        super().__init__(problem_type, is_biased_graph)
    
    def set_max_local_reward(self, spins: ArrayLike, matrix: ArrayLike) -> None:
        """
        The max local reward is for normalizing the solution quality in the inputs. 
        The max cut uses the maximum change in cut from an empty solution, but we'll use the maximum degree + number of nodes.
        This is to change the normalization depending on the graph topology.
        """

        ### op.matmul(matrix * spins, spins) gives the degree of each node
        self._max_local_reward = len(spins) + np.max(op.matmul(matrix * spins, spins))

    def set_invalidity_normalizer(self, spins: ArrayLike, matrix: ArrayLike) -> None:
        """
        To normalize the invalidity, we'll want to use the total number of edges in the graph.
        This is just the sum of every value in the matrix divided by 2, because it's an unweighted undirected graph.
        """

        self._invalidity_normalizer = np.sum(matrix) / 2

    def set_quality_normalizer(self, spins: ArrayLike, matrix: ArrayLike) -> None:
        """
        The solution quality can be at most the number of vertices, so we'll use this to normalize the quality.
        """
        self._solution_quality_normalizer = len(spins)

    def set_lower_bound(self, spins: ArrayLike, matrix: ArrayLike) -> None:
        self._lower_bound = 0

    def get_solution(self, spins: ArrayLike, matrix: ArrayLike) -> float:
        """
        The solution for the minimum vertex cover is just the size of the solution set. So sum of spins == 1.
        However, if it's an invalid solution, we'll set the solution to the number of vertices (worse solution)
        """
        if not self.is_valid(spins, matrix):
            return len(spins)
        
        return np.sum(spins == 1)
    
    def _get_measure(self, spins: ArrayLike, matrix: ArrayLike) -> float:
        """
        Same thing as solution but ignoring validity.
        """
        return np.sum(spins == 1)

    def get_solution_quality_mask(self, spins: ArrayLike, matrix: ArrayLike) -> NDArray:
        """
        For minimum vertex cover, the solution quality mask is the change in the number of nodes in the solution for every vertex.
        Because 1s are in and -1s are out of the solution, the solution quality change for each vertex is therefore just the 
        vertices themselves. -1 for adding a node to the solution, +1 for removing it.
        """
        return spins
    
    def get_invalidity_degree(self, spins: ArrayLike, matrix: ArrayLike) -> float:
        """
        The invalidity degree is the number of edges not covered by the solution. 
        """
        # np.array([spins == -1]) gives an array with 0s where there are nodes in the solution. Multiplying this and it's transpose
        # by the matrix gives the matrix with only uncovered edges (counted twice, as it'll be bi-directional due to undirected graph)
        # therefore sum and divide by two to get number of uncovered edges
        return np.sum(matrix * np.array([spins == -1]) * np.array([spins == -1]).T) / 2
    
    def get_invalidity_degree_mask(self, spins: ArrayLike, matrix: ArrayLike) -> NDArray:
        """
        Invalidity degree mask is the change in validity degree for every vertex spin, positive if it increases the invalidity,
        negative if it decreases the invalidity (i.e. it makes it closer to a valid solution).
        For the minimum vertex cover, this is the negative value for the number of newly covered edges.
        """
        # matrix * (spins == -1) gives a matrix but removing the edges incoming from any vertex in the solution on all vertices.
        # Doing matrix multiplication with the spins gives you negative values, representing the number of edges not covered
        # by a vertex in the solution (other than itself if it is in the solution) for each vertex, or 0 if there are no uncovered edges
        # Therefore by multiplying this new array by the vertices themselves, you get how many edges get covered on flipping that 
        # vertex, negative values indicating that it reduces the number of covered edges. For invalidity, positive values should decrease
        # invalidity, so multiply by -1 for correct values.
        return -1 * spins * op.matmul(matrix * np.array(spins == -1, dtype=np.float64), spins)
    
    def get_score_mask(self, spins: ArrayLike, matrix: ArrayLike) -> NDArray:
        """
        The score mask is the change in score for every vertex flip. We therefore need to compute the new quality and invalidity degree
        for each vertex flip.
        """

        # Solution quality on each flip is just the current quality + the mask
        # because solution quality ignores validity
        updated_quality = self.get_solution_quality(spins, matrix) + self.get_solution_quality_mask(spins, matrix)
        # Updated invalidity degree is just the current degree of invalidity + the mask
        # this gives 0s for creating valid solutions
        updated_invalidity = self.get_invalidity_degree(spins, matrix) + self.get_invalidity_degree_mask(spins, matrix)
        scores = self.get_validity_mask(spins, matrix) * updated_quality - updated_invalidity
        # Now we return the score on each vertex flip - the current score to get the update in score value on flip
        return scores - self.get_score(spins, matrix)
    
    def get_normalized_score_mask(self, spins: ArrayLike, matrix: ArrayLike) -> float:
        """
        Same calculations as the score mask, but normalize
        """

        updated_quality = self.get_solution_quality(spins, matrix) + self.get_solution_quality_mask(spins, matrix)
        updated_quality /= self._solution_quality_normalizer

        updated_invalidity = self.get_invalidity_degree(spins, matrix) + self.get_invalidity_degree_mask(spins, matrix)
        updated_invalidity /= self._invalidity_normalizer

        normalized_scores = self.get_validity_mask(spins, matrix) * updated_quality - updated_invalidity
        # Now we return the score on each vertex flip - the current score to get the update in score value on flip
        return normalized_scores - self.get_normalized_score(spins, matrix)



class MaximumCutUnbiasedScorer(MaximizationProblem):
    def __init__(self, problem_type, is_biased_graph):
        super().__init__(problem_type, is_biased_graph)

    def set_invalidity_normalizer(self, spins: ArrayLike, matrix: ArrayLike) -> None:
        """
        Maximum cut does not need this so we just set it to 1
        """
        self._invalidity_normalizer = 1

    def set_quality_normalizer(self, spins: ArrayLike, matrix: ArrayLike) -> None:
        """
        The total weight of positive vertex weights. In case of all negative weights, set to 1.
        """
        self._solution_quality_normalizer = max(1, np.sum(np.multiply(matrix, (matrix > 0))))

    def set_max_local_reward(self, spins: ArrayLike, matrix: ArrayLike) -> None:
        """
        For the maximum cut, the maximum local reward is the maximum immediate reward from the empty set 
        (i.e. the highest weight on any node). This is just calling the solution quality mask with an empty 
        (all -1) solution set. 
        """
        quality_mask = self.get_solution_quality_mask(np.array([-1] * len(spins), dtype=np.float64), matrix)
        without_zeros = quality_mask[np.nonzero(quality_mask)]
        self._max_local_reward = np.max(without_zeros)
    
    def get_solution(self, spins: ArrayLike, matrix: ArrayLike) -> float:
        """
        For any cut problem, the solution is just the quality of a maximum cut, i.e. the cut value
        """
        return calculate_cut(spins, matrix)
    
    def _get_measure(self, spins: ArrayLike, matrix: ArrayLike) -> float:
        """
        No validity with max cut, so just return the solution.
        """
        return self.get_solution(spins, matrix)

    def get_solution_quality_mask(self, spins: ArrayLike, matrix: ArrayLike) -> NDArray:
        """
        For the maximum cut, this will be the change in cut value
        for each spin flip compared to the passed spins.
        """
        return calculate_cut_changes(spins, matrix)

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
    
    def get_normalized_score_mask(self, spins: ArrayLike, matrix: ArrayLike) -> float:
        """
        For max cut, normalized score mask is the solution quality mask divided by the solution quality normalizer
        """
        return self.get_solution_quality_mask(spins, matrix) / self._solution_quality_normalizer



class MinimumCutUnbiasedSolver(MinimizationProblem):
    """
    The minimum cut requires all the same calculations as the maximum cut, so we'll just make it inherit and call super, then we 
    just flip the sign for all the scores to make it minimize instead of maximize. We can do this for basically any other problem.
    
    Because there's no invalidity, won't worry about offsets or anything like that.
    """
    def __init__(self, problem_type: OptimisationTarget, is_biased_graph: bool):
        super().__init__(problem_type, is_biased_graph)

    def set_invalidity_normalizer(self, spins: ArrayLike, matrix: ArrayLike) -> None:
        """
        Minimum cut does not need this so we just set it to 1
        """
        self._invalidity_normalizer = 1

    def set_quality_normalizer(self, spins: ArrayLike, matrix: ArrayLike) -> None:
        """
        The absolute total weight of negative vertex weights. In case of all positive weights, set to 1.
        """
        self._solution_quality_normalizer = max(1, abs(np.sum(np.multiply(matrix, (matrix < 0)))))

    def set_max_local_reward(self, spins: ArrayLike, matrix: ArrayLike) -> None:
        """
        For the minimum cut, the maximum local reward is the maximum immediate reward from the empty set 
        (i.e. the highest negative on any node). This is just calling the solution quality mask with an empty 
        (all -1) solution set. 
        """
        quality_mask = self.get_solution_quality_mask(np.array([-1] * len(spins), dtype=np.float64), matrix)
        without_zeros = quality_mask[np.nonzero(quality_mask)]
        self._max_local_reward = np.max(without_zeros)
    
    def get_solution(self, spins: ArrayLike, matrix: ArrayLike) -> float:
        """
        For any cut problem, the solution is just the quality of a maximum cut, i.e. the cut value
        """
        return calculate_cut(spins, matrix)
    
    def _get_measure(self, spins: ArrayLike, matrix: ArrayLike) -> float:
        """
        No validity with min cut, so just return the solution.
        """
        return self.get_solution(spins, matrix)

    def get_solution_quality_mask(self, spins: ArrayLike, matrix: ArrayLike) -> NDArray:
        """
        For the minimum cut, this will be the change in cut value for each spin flip compared to the passed spins, 
        but negative, because we want an increase for negative cut changes.
        """
        return - calculate_cut_changes(spins, matrix)

    def get_invalidity_degree(self, spins: ArrayLike, matrix: ArrayLike) -> float:
        """
        There is no such thing as an invalid solution for the min cut.
        Therefore we just set the invalidity degree to 0.
        """
        return 0
    
    def get_invalidity_degree_mask(self, spins: ArrayLike, matrix: ArrayLike) -> NDArray:
        """
        All min cut solutions are valid. Therefore the invalidity degree is all 0s.
        """
        return [0 for _ in range(len(spins))]
    
    def get_score_mask(self, spins: ArrayLike, matrix: ArrayLike) -> NDArray:
        """
        Score mask for the case of minimum cut is the same as the solution quality mask.
        """
        return self.get_solution_quality_mask(spins, matrix)
    
    def get_normalized_score_mask(self, spins: ArrayLike, matrix: ArrayLike) -> float:
        """
        Normalized score mask is the solution quality mask divided by the solution quality normalizer
        """
        return self.get_solution_quality_mask(spins, matrix) / self._solution_quality_normalizer
    


class MaximumIndependentSetUnbiasedSolver(MaximizationProblem):
    def __init__(self, problem_type: OptimisationTarget, is_biased_graph: bool):
        super().__init__(problem_type, is_biased_graph)

    def set_invalidity_normalizer(self, spins: ArrayLike, matrix: ArrayLike) -> None:
        """
        Invalidity is measured the same way as the minimum vertex cover. Take the number of edges.
        """
        self._invalidity_normalizer = np.sum(matrix) / 2

    def set_max_local_reward(self, spins: ArrayLike, matrix: ArrayLike) -> None:
        """
        Once again, the same as the minimum vertex cover. Take the maximum degree + the number of vertices in the graph.
        """
        self._max_local_reward = len(spins) + np.max(op.matmul(matrix * spins, spins))

    def set_lower_bound(self, spins: ArrayLike, matrix: ArrayLike) -> None:
        """
        The lower bound for this problem is always 0 because it's based on the number of vertices which is never negative.
        """
        self._lower_bound = 0

    def set_quality_normalizer(self, spins: ArrayLike, matrix: ArrayLike) -> None:
        """
        As the solution quality is based on the number of vertices in S, the maximum is the number of vertices in G.
        """
        self._solution_quality_normalizer = len(spins)

    def get_solution(self, spins: ArrayLike, matrix: ArrayLike) -> float:
        """
        Returns the number of vertices in the solution. If the solution is invalid, return none.
        """
        if not self.is_valid(spins, matrix):
            return 0
        
        return np.sum(spins == 1)
    
    def get_solution_quality_mask(self, spins: ArrayLike, matrix: ArrayLike) -> NDArray:
        """
        Seeing as it's a maximization problem, spins not in the solution (-1 state) should give a +1 quality and those in
        (1 state) should give a -1 quality. This is just negative of the spins.
        """
        return - spins
    
    def get_invalidity_degree(self, spins: ArrayLike, matrix: ArrayLike) -> float:
        """
        Calculating invalidity is similar to the vertex cover. By getting rid of all edges not covered by the solution,
        we get the set of edges inside the solution set. Counting these gives the invalidity.
        """

        ### The spins == 1 gives an array with 0s in place of vertices not in the solution. By multiplying the matrix by this and 
        ### the transpose, you get an adjacency matrix containing only the edges that connect vertices in the solution. Because
        ### the graph is undirected and unweighted, all of these edges would be counted twice if you added them all up, so divide by two.
        return np.sum(matrix * np.array([spins == 1]) * np.array([spins == 1]).T) / 2
    
    def get_invalidity_degree_mask(self, spins: ArrayLike, matrix: ArrayLike) -> NDArray:
        """
        Very similar calculation to the minimum vertex cover, but instead of counting edges that are not incident on a vertex in the solution,
        we instead count edges that are incident on a vertex in the solution.
        """

        ### Where matrix * spins == -1 gave the edges that are not incident on the solution, matrix * spins == 1 gives the ones that are,
        ### only counted once. By matrix multiplying with the spins themselves, you get the count of edges per vertex that connect it to
        ### a vertex in the solution. By then multiplying normally by the spins, you get the negative change in the number of edges 
        ### connected within the solution if that vertex is flipped. That is, positive if reducing the amount of interconnected edges in
        ### the solution, and negative if increasing the amount of interconnecting edges.
        ### Therefore, to get an invalidity degree, which is positive the worse the solution gets, we want to flip the sign, in order to get
        ### increasing values for more edges contained within the solution set.
        return - spins * op.matmul(matrix * np.array(spins == 1, dtype=np.float64), spins)
    
    def get_score_mask(self, spins: ArrayLike, matrix: ArrayLike) -> NDArray:
        """
        The score mask is the change in score for every vertex flip. We therefore need to compute the new quality and invalidity degree
        for each vertex flip. This is done identically to the Minimum Vertex Cover.
        """

        # Solution quality on each flip is just the current quality + the mask
        # because solution quality ignores validity
        updated_quality = self.get_solution_quality(spins, matrix) + self.get_solution_quality_mask(spins, matrix)
        # Updated invalidity degree is just the current degree of invalidity + the mask
        # this gives 0s for creating valid solutions
        updated_invalidity = self.get_invalidity_degree(spins, matrix) + self.get_invalidity_degree_mask(spins, matrix)
        scores = self.get_validity_mask(spins, matrix) * updated_quality - updated_invalidity
        # Now we return the score on each vertex flip - the current score to get the update in score value on flip
        return scores - self.get_score(spins, matrix)
    
    def get_normalized_score_mask(self, spins: ArrayLike, matrix: ArrayLike) -> float:
        """
        Same calculations as the score mask, but normalizing the quality and invalidity.
        """

        updated_quality = self.get_solution_quality(spins, matrix) + self.get_solution_quality_mask(spins, matrix)
        updated_quality /= self._solution_quality_normalizer

        updated_invalidity = self.get_invalidity_degree(spins, matrix) + self.get_invalidity_degree_mask(spins, matrix)
        updated_invalidity /= self._invalidity_normalizer

        normalized_scores = self.get_validity_mask(spins, matrix) * updated_quality - updated_invalidity
        # Now we return the score on each vertex flip - the current score to get the update in score value on flip
        return normalized_scores - self.get_normalized_score(spins, matrix)

    

class ScoreSolverFactory():
    @staticmethod
    def get(problem_type : env_utils.OptimisationTarget, is_biased_graph : bool) -> ScoreSolver:
        """
        Creates a ScoreSolver depending on the optimization target and whether the systems are biased (i.e. digraphs)
        """
        if problem_type == env_utils.OptimisationTarget.CUT and not is_biased_graph:
            return MaximumCutUnbiasedScorer(problem_type, is_biased_graph)
        
        elif problem_type == env_utils.OptimisationTarget.MIN_COVER and not is_biased_graph:
            return MinimumVertexCoverUnbiasedScorer(problem_type, is_biased_graph)
        
        elif problem_type == env_utils.OptimisationTarget.MIN_CUT and not is_biased_graph:
            return MinimumCutUnbiasedSolver(problem_type, is_biased_graph)
        
        elif problem_type == env_utils.OptimisationTarget.MAX_IND_SET and not is_biased_graph:
            return MaximumIndependentSetUnbiasedSolver(problem_type, is_biased_graph)
        
        # If make it here, invalid target
        raise NotImplementedError("Invalid optimization target: %s and biased %s", problem_type, is_biased_graph)