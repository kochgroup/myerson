from myerson import ShapleyCalculator, ShapleySampler

try:
    import torch
    import torch_geometric
except ImportError:
    raise ImportError("Failed to import torch and/or torch_geometric. PyG explanations not available.")
import numpy as np

import networkx as nx
from tqdm import tqdm
import logging


class ShapleyExplainer(ShapleyCalculator):
    r"""Explains the prediction of a graph neural network (GNN) with Shapley values.
        The GNN is treated as the coalition function of a game and its prediction
        as the payoff of the game. The Shapley values show how much each node of 
        the graph contributed to the final prediction.

    Args:
        graph (torch_geometric.data.Data): The data instance that is to be explained.
        coalition_function (torch.nn.Module): The GNN.
        disable_tqdm (bool, optional): Disables progress bar. Defaults to True.
    """

    def __init__(self, 
                graph: torch_geometric.data.Data,
                coalition_function: torch.nn.Module,
                disable_tqdm: bool=True) -> None:
        """Instantiate the class.
        """

        self.disable_tqdm = disable_tqdm
        self.log = logging.getLogger("ShapleyExplainer")

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.log.info(f"using device {self.device}")
        self.pyg_graph = graph.to(self.device)
        self.coalition_function = coalition_function
        self.coalition_function.to(self.device)

        self.nx_graph = torch_geometric.utils.to_networkx(graph, to_undirected=True)
        self.grand_coalition = list(self.nx_graph.nodes()) # alias: set of players / set of nodes / F

    def calculate_worth_of_single_coalition(self,
        coalition: tuple,
        pyg_graph: torch_geometric.data.Data) -> float:
        """Calculate the worth of a coalition, i. e. a collection of nodes.

        Args:
            coalition (tuple): Coalition as node indices.
            pyg_graph (torch_geometric.data.Data): Graph from which a subgraph
                will be extracted. For the Shapley values, disconnected graphs
                are allowed.

        Returns:
            float: Worth, the output of the coalition function for the subgraph. 
        """
        if not coalition:
            return 0.
        subgraph = self.subgraph_from_coalition(coalition, pyg_graph)
        out = self.coalition_function(subgraph.x, subgraph.edge_index, self._batch_var(subgraph))
        
        return out.cpu().item()

    def calculate_worth_of_coalitions(self,
        coalitions: list) -> dict:
        """Calculate the worth of every coalition and map it to its worth. 

        Args:
            coalitions (list): Set of connected components as tuples of node indices.

        Returns:
            dict: Dictionary mapping each connected component to its worth.
        """
        self.log.info(f"Calculating worth of coalitions.")
        coalitions_to_worth = {}
        for coalition in tqdm(coalitions,
                            desc="Calculating worth of coalitions",
                            disable=self.disable_tqdm):
            worth = self.calculate_worth_of_single_coalition(coalition, self.pyg_graph)
            coalitions_to_worth.update({coalition: worth})
        return coalitions_to_worth

    def calculate_prediction(self) -> float:
        """Calculate the prediction of the GNN for the investigated graph.

        Returns:
            float: Prediction.
        """
        return self.coalition_function(self.pyg_graph.x, self.pyg_graph.edge_index,
                                    self._batch_var(self.pyg_graph)).cpu().item()

    def _batch_var(self, pyg_graph: torch_geometric.data.Data) -> torch.tensor:
        """Return a batch argument for single graphs, required for models 
        trained in batches.

        Args:
            pyg_graph (torch_geometric.data.Data): Graph for which to generate
                batch.

        Returns:
            torch.tensor: Batch attribute in the correct dimensions.
        """
        return torch.zeros(pyg_graph.x.shape[0], dtype=int, device=pyg_graph.x.device)

    def subgraph_from_coalition(self, coalition: tuple, 
                                pyg_graph: torch_geometric.data.Data) -> torch_geometric.data.Data:
        """Generates a subgraph from a coalition (a subset of nodes / players)
            and a graph. May return disconnected graphs.

        Args:
            nodes (tuple): Nodes which form the subgraph.
            pyg_graph (torch_geometric.data.Data): Subgraph induced in this
                graph by the subset of nodes.

        Returns:
            torch_geometric.data.Data: The new subgraph.
        """
        # unsorted nodes can result in the wrong edges
        nodes = sorted(coalition)
        nodes = torch.tensor(nodes, dtype=torch.long, device=pyg_graph.x.device)
        node_mask = torch.zeros(pyg_graph.x.shape[0], dtype=torch.bool, device=pyg_graph.x.device)
        node_mask[nodes] = True
        x = pyg_graph.x[node_mask]

        edge_mask = node_mask[pyg_graph.edge_index[0]] & node_mask[pyg_graph.edge_index[1]]
        edge_index = pyg_graph.edge_index[:, edge_mask]
        # fancy indexing to relabel edge_index
        node_idx = torch.zeros(node_mask.size(0), dtype=torch.long, device=pyg_graph.x.device)
        node_idx[nodes] = torch.arange(node_mask.sum().item(), device=pyg_graph.x.device)
        edge_index = node_idx[edge_index]

        subgraph = torch_geometric.data.Data(x=x, edge_index=edge_index)
        return subgraph

class ShapleySamplingExplainer(ShapleySampler, ShapleyExplainer):
    """A class explaining GNN predictions with approximated Shapley values.

    Args:
        graph (torch_geometric.data.Data): The data instance that is to be explained.
        coalition_function (torch.nn.Module): The GNN.
        seed (None | int, optional): Seed for randomness. Defaults to None.
        number_of_samples (int, optional): Number of sampling steps. Defaults to 1000.
        disable_tqdm (bool, optional): Disables progress bar. Defaults to True.
    """
    def __init__(self,
                graph: torch_geometric.data.Data,
                coalition_function: torch.nn.Module,
                seed: None | int = None, 
                number_of_samples: int = 1000,
                disable_tqdm: bool=True) -> None:
        """Instantiates the class.
        """
        self.disable_tqdm = disable_tqdm
        self.log = logging.getLogger("ShapleySamplingExplainer")

        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.number_of_samples = number_of_samples

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.log.info(f"using device {self.device}")
        self.pyg_graph = graph.to(self.device)
        self.coalition_function = coalition_function
        self.coalition_function.to(self.device)

        self.nx_graph = torch_geometric.utils.to_networkx(graph, to_undirected=True)
        self.grand_coalition = list(self.nx_graph.nodes()) # alias: set of players / set of nodes / F

class ShapleyClassExplainer(ShapleyExplainer):
    r"""Explains the prediction of a graph neural network (GNN) classifier with
        Shapley values.  The GNN is treated as the coalition function of a game
        and its prediction as the payoff of the game. The Shapley values show
        how much each node of the graph contributed to the final prediction.

    Args:
        graph (torch_geometric.data.Data): The data instance that is to be explained.
        coalition_function (torch.nn.Module): The GNN.
        disable_tqdm (bool, optional): Disables progress bar. Defaults to True.
    """

    def __init__(self, 
                graph: torch_geometric.data.Data,
                coalition_function: torch.nn.Module,
                disable_tqdm: bool=True) -> None:
        """Instantiate the class.
        """

        self.disable_tqdm = disable_tqdm
        self.log = logging.getLogger("ShapleyClassExplainer")

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.log.info(f"using device {self.device}")
        self.pyg_graph = graph.to(self.device)
        self.coalition_function = coalition_function
        self.coalition_function.to(self.device)

        self.nx_graph = torch_geometric.utils.to_networkx(graph, to_undirected=True)
        self.grand_coalition = list(self.nx_graph.nodes()) # alias: set of players / set of nodes / F
        self.pred = self.calculate_prediction()

    def calculate_worth_of_single_coalition(self, 
        coalition: tuple,
        pyg_graph: torch_geometric.data.Data) -> torch.tensor:
        """Calculate the worth of a coalition, i. e. a collection of nodes.

        Args:
            coalition (tuple): Coalition as node indices.
            pyg_graph (torch_geometric.data.Data): Graph from which a subgraph
                will be extracted. For the Shapley values, disconnected graphs
                are allowed.

        Returns:
            tensor: Worth, the output of the coalition function for the subgraph. 
        """
        if not coalition:
            return torch.zeros(self.pred.shape)
        subgraph = self.subgraph_from_coalition(coalition, pyg_graph)
        out = self.coalition_function(subgraph.x, subgraph.edge_index, self._batch_var(subgraph))
        
        return out.detach().cpu().squeeze(0)

    def calculate_prediction(self) -> torch.tensor:
        """Calculate the prediction of the GNN for the investigated graph.

        Returns:
            float: Prediction.
        """
        return self.coalition_function(self.pyg_graph.x, self.pyg_graph.edge_index,
                                    self._batch_var(self.pyg_graph)).cpu().squeeze(0)


class ShapleySamplingClassExplainer(ShapleySamplingExplainer, ShapleyClassExplainer):
    """A class explaining a GNNs classifier predictions with approximated Shapley values.

    Args:
        graph (torch_geometric.data.Data): The data instance that is to be explained.
        coalition_function (torch.nn.Module): The GNN.
        seed (None | int, optional): Seed for randomness. Defaults to None.
        number_of_samples (int, optional): Number of sampling steps. Defaults to 1000.
        disable_tqdm (bool, optional): Disables progress bar. Defaults to True.
    """
    def __init__(self,
                graph: torch_geometric.data.Data,
                coalition_function: torch.nn.Module,
                seed: None | int = None, 
                number_of_samples: int = 1000,
                disable_tqdm: bool=True) -> None:
        """Instantiates the class.
        """
        self.disable_tqdm = disable_tqdm
        self.log = logging.getLogger("ShapleySamplingClassExplainer")

        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.number_of_samples = number_of_samples

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.log.info(f"using device {self.device}")
        self.pyg_graph = graph.to(self.device)
        self.coalition_function = coalition_function
        self.coalition_function.to(self.device)

        self.nx_graph = torch_geometric.utils.to_networkx(graph, to_undirected=True)
        self.grand_coalition = list(self.nx_graph.nodes()) # alias: set of players / set of nodes / F
        self.pred = self.calculate_prediction()
    
    def sample_all_shapley_values(self) -> np.ndarray:
        """Use Monte Carlo sampling to approximate the Shapley values for every
        node / player in the graph.

        Returns:
            np.ndarray: Sampled Shapley values.
        """
        self.sample_all_mappings()
        pred = self.calculate_prediction()
        nodes_array = np.array(self.grand_coalition)
        sh_values = np.zeros((len(nodes_array), pred.shape[0]), dtype=float)
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
                sh_values[node_idx] = (sh_values[node_idx] + worth_with_node.numpy().squeeze() - worth_without_node.numpy().squeeze())

        sh_values = sh_values / self.number_of_samples
        log_string = "".join([f"\t{node}: {val}\n" for node, val in zip(self.grand_coalition, sh_values)])
        self.log.info(f"Sampled Shapley Values:\n{log_string}")
        return sh_values
