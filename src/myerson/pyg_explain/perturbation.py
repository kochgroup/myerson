try:
    import torch
    import torch_geometric
except ImportError:
    raise ImportError("Failed to import torch and/or torch_geometric. PyG explanations not available.")
import numpy as np
from tqdm import tqdm
import logging

class PerturbationExplainer():
    r"""Docstring...

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
        self.log = logging.getLogger("PerturbationExplainer")

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.log.info(f"using device {self.device}")
        self.pyg_graph = graph.to(self.device)
        self.coalition_function = coalition_function
        self.coalition_function.to(self.device)

        self.nx_graph = torch_geometric.utils.to_networkx(graph, to_undirected=True)
        self.grand_coalition = list(self.nx_graph.nodes()) # alias: set of players / set of nodes / F
        self.pred = self.calculate_prediction()

    def calculate_prediction(self) -> torch.tensor:
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

    def calculate_all_perturbation_values(self) -> np.ndarray:
        """Calculate the perturbation values for every node / player in the graph.
           The explanations are calculated as the difference between the worth of 
           the grand coalition and the worth of the graph without the node of interest.

        Returns:
            np.ndarray: Perturbation values.
        """
        self.log.info(f"Calculating perturbation values.")
        perturbation_values = []
        with torch.no_grad():
            for i, node in enumerate(tqdm(self.grand_coalition, desc="Calculating perturbation values.", disable=self.disable_tqdm)):
                coalition = tuple(self.grand_coalition[:i] + self.grand_coalition[i+1:])
                worth_of_coalition = self.calculate_worth_of_single_coalition(coalition, self.pyg_graph)
                pert_val = (self.pred - worth_of_coalition)
                perturbation_values.append(pert_val)
            perturbation_values = np.array(perturbation_values)
            log_string = "".join([f"\t{node}: {val}\n" for node, val in zip(self.grand_coalition, perturbation_values)])
            self.log.info(f"Perturbation Values:\n{log_string}")
        return perturbation_values

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
        
        return out.cpu().item()

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

class PerturbationClassExplainer(PerturbationExplainer):
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
        super().__init__(graph, coalition_function, disable_tqdm)

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