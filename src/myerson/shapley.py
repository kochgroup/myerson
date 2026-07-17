import math
import networkx as nx
import numpy as np
from itertools import combinations, chain

from typing import Callable 

from tqdm import tqdm
import logging


class ShapleyCalculator():
    r"""Calculates the exact Shapley values. 
        For a game described by a coalition function :math:`v` the Shapley values
        attribute the individual players contribution to the payoff of the game
        (:math:`S`: coalition of players, :math:`N`: grand coalition of all players):

        .. math::

            \text{Sh}_i\,({v}) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! \: (|N| - |S| - 1)!}{|N|!}\big( {v}\,(S \cup \{i\}) - {v}\,(S) \big)

    Args:
        graph (nx.classes.graph.Graph): The players of the game. The graph
            structure (edges) are not used for the Shapley value.
        coalition_function (Callable): The coalition for which to calculate
            the payoff of the game. Expects a coalition (tuple of node
            indices) and a graph which contains additional information on
            the players to decide on the payoff. 

        disable_tqdm (bool, optional): Disables progress bar. Defaults to
            True.
    """

    def __init__(self,
                 graph: nx.classes.graph.Graph,
                 coalition_function: Callable,
                 disable_tqdm: bool=True) -> None:
        """Instantiate the class.
        """

        self.disable_tqdm = disable_tqdm
        self.log = logging.getLogger("ShapleyCalculator")
        self.nx_graph = graph
        self.grand_coalition = list(graph.nodes()) # alias: set of players / set of nodes / F
        self.coalition_function = coalition_function

    def calculate_coalitions(self, grand_coalition: list) -> list:
        r"""Calculate all possible coalitions for a set of players / all nodes in
        a graph.

        Args:
            grand_coalition (list): Set of players / grand coalition / atoms in
                graph as a list of tupels.

        Returns:
            list: All :math:`2^N` coalitions.
        """
        self.log.info(f"Calculating number of coalitions.")
        coalitions = [combinations(grand_coalition, len(grand_coalition)-x) for x in \
                      tqdm(range(len(grand_coalition)), desc="Calculate coalitions", disable=self.disable_tqdm)]
        coalitions = list(chain.from_iterable(coalitions)) # chaining removes empty set
        coalitions.append(())
        self.log.info(f"Number of coalitions: {len(coalitions)}")
        return coalitions

    def calculate_worth_of_single_coalition(self,
        coalition: tuple,
        nx_graph: nx.classes.graph.Graph) -> float:
        """Calculate the worth of a singel coalition.

        Args:
            coalition (tuple): Coalition as node indices.
            nx_graph (nx.classes.graph.Graph): Additional information for the
                coalition function, i.e. the entire graph with node parameters. 
                The result of the coalition function should depend only on the
                coalition, however the node parameters might contain necessary 
                information.

        Returns:
            float: Worth, the output of the coalition function for the players in
                coalition.
        """
        return self.coalition_function(coalition, nx_graph)

    def calculate_worth_of_coalitions(self, coalitions: list) -> dict:
        """Calculate the worth of every coalition and map it to its worth. 

        Args:
            coalitions (list): Set of coalitions as tuples of node indices.

        Returns:
            dict: Dictionary mapping each coalition to its worth.
        """
        self.log.info(f"Calculating worth of coalitions.")
        coalitions_to_worth = {}
        for coalition in tqdm(coalitions,
                              desc="Calculating worth of coalitions",
                              disable=self.disable_tqdm):
            worth = self.calculate_worth_of_single_coalition(coalition, self.nx_graph)
            coalitions_to_worth.update({coalition: worth})
        return coalitions_to_worth

    def calculate_worth_of_grand_coalition(self, nx_graph: nx.classes.graph.Graph) -> float:
        """Calculate payoff of the game, i.e. the payoff of all players / the
        grand coalition.

        Args:
            coalition_function (Callable): The coalition function associating a
                coalition with a payoff.
            nx_graph (nx.classes.graph.Graph): The players are nodes in a graph.

        Returns:
            float: Payoff of the game / worth of grand coalition.
        """
        return self.coalition_function(self.grand_coalition, nx_graph)

    def _precompute_prefactors(self, size_grand_coalition: int) -> list[float]:
        """Precompute Shapley prefactors for each coalition size.

        Args:
            size_grand_coalition (int): Number of players.

        Returns:
            list[float]: Prefactor for coalition size s at index s.
        """
        n_fact = math.factorial(size_grand_coalition)
        return [
            math.factorial(s) * math.factorial(size_grand_coalition - s - 1) / n_fact
            for s in range(size_grand_coalition)
        ]

    def calculate_single_shapley_value(self, node: int, grand_coalition: tuple,
                                  coalitions: list[tuple], coalition_to_worth: dict) -> float:
        """Calculate a single Shapley value.

        Args:
            node (int): Node index for which to calculate the Shapley value.
            grand_coalition (tuple): Set of all players.
            coalitions (list[tuple]): List of all coalitions.
            coalition_to_worth (dict): Mapping of every coalition to its worth.

        Returns:
            float: Shapley value.
        """
        sh = 0
        size_grand_coalition = len(grand_coalition)
        prefactors = self._precompute_prefactors(size_grand_coalition)
        for coalition in coalitions:
            if node in coalition:
                continue
            size_coalition = len(coalition)
            worth_of_coalition = coalition_to_worth[coalition]
            worth_of_coalition_with_node = coalition_to_worth[tuple(sorted(coalition+(node,)))]
            sh += prefactors[size_coalition] * (worth_of_coalition_with_node - worth_of_coalition)
        return sh

    def calculate_all_shapley_values(self) -> np.ndarray:
        """Calculate the Shapley values for every node / player in the graph.

        Returns:
            np.ndarray: Shapley values for each node.
        """
        self.calculate_all_mappings()
        self.log.info(f"Calculating Shapley values.")
        sh_values = []
        for node in tqdm(self.grand_coalition, desc="Calculating Shapley values.", disable=self.disable_tqdm):
            sh_val = self.calculate_single_shapley_value(node, self.grand_coalition,
                                                  self.coalitions, self.coalitions_to_worth)
            sh_values.append(sh_val)
        sh_values = np.array(sh_values)
        log_string = "".join([f"\t{node}: {val}\n" for node, val in zip(self.grand_coalition, sh_values)])
        self.log.info(f"Shapley Values:\n{log_string}")
        return sh_values

    def calculate_all_mappings(self) -> None:
        """Calculates all coalitions, and their associated worths as class attributes:

            * `self.coalitions` (list[tuple])
            * `self.coalitions_to_worth` (dict)
        """
        self.coalitions = self.calculate_coalitions(self.grand_coalition)

        self.coalitions_to_worth \
            = self.calculate_worth_of_coalitions(self.coalitions)

class ShapleySampler(ShapleyCalculator):
    r"""A class approximating the Shapley value using Monte Carlo sampling.
        Approximated by randomly sampling from all  permutations needed to
        calculate the Shapley value:

        .. math::

            \text{Sh}_i\,({v}) = \frac{1}{|N|!}\; \sum_R \big({v}\,(P_i^R \cup \{i\}) - {v}(P_i^R)\big)

        For efficiencies sake, the sampled permutations are transformed into
        the corresponding coalitions.

    Args:
        graph (nx.classes.graph.Graph): The players of the game. The graph
            structure (edges) are not used for the Shapley value.
        coalition_function (Callable): The coalition for which to calculate
            the payoff of the game. Expects a coalition (tuple of node
            indices) and a graph which contains additional information on
            the players to decide on the payoff. 
        seed (None | int, optional): Seed for randomness. Defaults to None.
        number_of_samples (int, optional): Number of sampling steps. Defaults to 1000.
        disable_tqdm (bool, optional): Disables progress bar. Defaults to True.
    """
    def __init__(self,
                 graph: nx.classes.graph.Graph,
                 coalition_function: Callable,
                 seed: None | int = None, 
                 number_of_samples: int = 1000,
                 disable_tqdm: bool=True) -> None:

        self.disable_tqdm = disable_tqdm
        self.log = logging.getLogger("ShapleySampler")
        self.nx_graph = graph
        self.grand_coalition = list(graph.nodes()) # alias: set of players / set of nodes / F
        self.coalition_function = coalition_function
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.number_of_samples = number_of_samples
        """Instantiates the class.
        """

    @staticmethod
    def _replace_in_array(array: np.ndarray, value_to_replace, value_to_replace_with) -> np.ndarray:
        """Replace a value in an array with a different value.

        Args:
            array (np.ndarray): The array.
            value_to_replace (int): The value to replace.
            value_to_replace_with (int): The value to replace with.

        Returns:
            np.ndarray: The array with the replace value if it was found, else
                the original array.
        """
        # Convert to a flat array to work with a single loop for any dimension
        flat_array = array.ravel().copy()
        # Find the index of the first occurrence of value_to_replace
        index = np.where(flat_array == value_to_replace)[0]
        if index.size > 0:  # Check if the value was found
            flat_array[index[0]] = value_to_replace_with  # Replace the first occurrence
        # Reshape back to original array's shape
        return flat_array.reshape(array.shape)

    def reset_rng(self):
        """Reset random number generator to seed.
        """
        self.rng = np.random.default_rng(self.seed)

    def sample_permutations(self, number_of_samples: int):
        """Uniformly sample permutations from all possible permutations. Samples
        `number_of_samples`*2*`number_of_nodes` permutations in total.

        Args:
            number_of_samples (int): How many sample steps should be carried out.

        Returns:
            tuple(int, list[np.ndarray], list[np.ndarray]): A randomly chosen 
                node, the sampled permutations without the random node, all the
                sampled permutations.
        """
        nodes_array = np.array(self.grand_coalition)
        random_node = self.rng.choice(nodes_array)

        self.log.info(f"Sampling {number_of_samples} steps.")
        permutations_without_random_node = []
        for i in tqdm(range(number_of_samples),
                      desc="Sample permutations without random node",
                      disable=self.disable_tqdm):
            random_permutation_size: int = self.rng.integers(0, len(nodes_array))  
            nodes_array_without_random_node = nodes_array.copy()
            indices_to_delete = np.where(nodes_array_without_random_node == random_node)
            nodes_array_without_random_node = np.delete(nodes_array_without_random_node,
                                                        indices_to_delete)
            sampled_permutation_without_random_node = self.rng.choice(nodes_array_without_random_node,
                                                        size=random_permutation_size,
                                                        replace=False)
            self.rng.shuffle(sampled_permutation_without_random_node)
            permutations_without_random_node.append(sampled_permutation_without_random_node)

        all_sampled_permutations = []
        for permutation in tqdm(permutations_without_random_node,
                         desc="Sample permutations containing random node",
                         disable=self.disable_tqdm):
            for node_idx, node in enumerate(nodes_array):
                sampled_permutation_with_current_swapped_in_random_node = permutation.copy()
                sampled_permutation_with_current_swapped_in_random_node = \
                self._replace_in_array(sampled_permutation_with_current_swapped_in_random_node,
                                       node, random_node)
                all_sampled_permutations.append(sampled_permutation_with_current_swapped_in_random_node)
                all_sampled_permutations.append(np.append(sampled_permutation_with_current_swapped_in_random_node, node))
        # len(all_sampled_permutations): steps*2*len(self.grand_coalition)
        self.log.info(f"Sampled {len(all_sampled_permutations)} of {math.factorial(len(self.grand_coalition))} permutations.")

        return random_node, permutations_without_random_node, all_sampled_permutations

    def get_coalitions_from_permutations(self, permutations: list[np.ndarray]):
        """Get the set of coalitions from the different (sampled) permutations.

        Args:
            permutations (list[np.ndarray]): The permutations.

        Returns:
            list[tuple]: The sampled coalitions.
        """
        all_sampled_coalitions = list(set([tuple(np.sort(x)) for x in permutations]))
        self.log.info(f"Sampled {len(all_sampled_coalitions)} of {2**len(self.grand_coalition)} coalitions.")
        return all_sampled_coalitions

    def sample_all_mappings(self) -> None:
        """Samples permutations, corresponding coalitions and their associated
            worths as class attributes:

            * `self.random_node` (int) 
            * `self.permutations_without_random_node` (list[np.ndarray])
            * `self.all_sampled_permutations` (list[np.ndarray])
            * `self.coalitions` (list[tuple])
            * `self.coalitions_to_worth` (dict)
        """
        self.random_node, self.permutations_without_random_node, self.all_sampled_permutations \
            = self.sample_permutations(self.number_of_samples)

        self.coalitions = self.get_coalitions_from_permutations(self.all_sampled_permutations)

        self.coalitions_to_worth = self.calculate_worth_of_coalitions(self.coalitions)

    def sample_all_shapley_values(self) -> np.ndarray:
        """Use Monte Carlo sampling to approximate the Shapley values for every
        node / player in the graph.

        Returns:
            np.ndarray: Sampled Shapley values.
        """
        self.sample_all_mappings()
        nodes_array = np.array(self.grand_coalition)
        sh_values = np.zeros(len(nodes_array), dtype=float)
        self.log.info(f"Calculating sampled Shapley values.")
        for permutation in tqdm(self.permutations_without_random_node,
                              disable=self.disable_tqdm,
                              desc="Calculate sampled Shapley values"):
            for node_idx, node in enumerate(nodes_array):

                sampled_permutation_with_current_swapped_in_random_node = permutation.copy()
                sampled_permutation_with_current_swapped_in_random_node \
                    = self._replace_in_array(sampled_permutation_with_current_swapped_in_random_node,
                                             node,
                                             self.random_node)

                worth_with_node = self.coalitions_to_worth[tuple(np.sort(np.append(sampled_permutation_with_current_swapped_in_random_node, node)))]
                worth_without_node = self.coalitions_to_worth[tuple(np.sort(sampled_permutation_with_current_swapped_in_random_node))]
                sh_values[node_idx] = (sh_values[node_idx] + worth_with_node - worth_without_node)

        sh_values = sh_values / self.number_of_samples
        log_string = "".join([f"\t{node}: {val:.4f}\n" for node, val in zip(self.grand_coalition, sh_values)])
        self.log.info(f"Sampled Shapley Values:\n{log_string}")
        return sh_values

    def calculate_all_shapley_values(self) -> None:
        """Not implemented for sampling class.

        Raises:
            NotImplementedError: The ShapleySampler only has the
                `sample_all_shapley_values()` method. To accuratly calculate the
                Shapley values, please use the `ShapleyCalculator` class.
        """
        raise NotImplementedError("""The ShapleySampler only has the `sample_all_shapley_values()` method.
                     To accuratly calculate the Shapley values,
                     please use the `ShapleyCalculator` class.""")

    def calculate_single_shapley_value_with_sampling(self, node: int, grand_coalition: tuple,
                                  coalitions: list[tuple], coalition_to_worth: dict) -> float:
        """Calculate a single Shapley value using sampled coalitions. This leads
            to worse results than the permutation based approached unless a large
            portion of the coalitions is sampled.

        Args:
            node (int): Node index for which to calculate the Shapley value.
            grand_coalition (tuple): Set of all players.
            coalitions (list[tuple]): List of all coalitions.
            coalition_to_worth (dict): Mapping of every coalition to its worth.

        Returns:
            float: Shapley value.
        """
        # TODO: look for potential improvement
        sh = 0
        size_grand_coalition = len(grand_coalition)
        prefactors = self._precompute_prefactors(size_grand_coalition)
        for coalition in coalitions:
            if node in coalition:
                continue
            coalition_with_node = tuple(sorted(coalition+(node,)))
            if coalition_with_node in coalitions:
                worth_of_coalition = coalition_to_worth[coalition]
                worth_of_coalition_with_node = coalition_to_worth[coalition_with_node]
                sh += prefactors[len(coalition)] * (worth_of_coalition_with_node - worth_of_coalition)
        return sh

    def calculate_all_shapley_values_with_sampling(self) -> np.ndarray:
        """Calculate the sampled Shapley values for every node / player in the
            graph.  This leads to worse results than the permutation based
            approached unless a large portion of the coalitions is sampled.


        Returns:
            np.ndarray: Shapley values.
        """
        self.sample_all_mappings()
        self.log.warning(f"This sampling method performs worse than `sample_all_myerson_values` and might not sample uniformly.")
        self.log.info(f"Calculating Shapley values.")
        sh_values = []
        for node in tqdm(self.grand_coalition, desc="Calculating Shapley values.", disable=self.disable_tqdm):
            sh_val = self.calculate_single_shapley_value(node, self.grand_coalition,
                                                  self.coalitions, self.coalitions_to_worth)
            sh_values.append(sh_val)
        sh_values = np.array(sh_values)
        log_string = "".join([f"\t{node}: {val}\n" for node, val in zip(self.grand_coalition, sh_values)])
        self.log.info(f"Shapley Values:\n{log_string}")
        return sh_values