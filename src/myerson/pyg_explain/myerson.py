from myerson import MyersonCalculator, MyersonSampler

try:
    import torch
    import torch_geometric
except ImportError:
    raise ImportError("Failed to import torch and/or torch_geometric. PyG explanations not available.")
import numpy as np

from tqdm import tqdm
import logging
# try:
#     from .myerson import fast_restrict
# except:
#     pass


class MyersonExplainer(MyersonCalculator):
    r"""Explains the prediction of a graph neural network (GNN) with Myerson values.
        The GNN is treated as the coalition function of a game and its prediction
        as the payoff of the game. The Myerson values show how much each node of 
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
        self.log = logging.getLogger("MyersonExplainer")

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.log.info(f"using device {self.device}")
        self.pyg_graph = graph.to(self.device)
        self.coalition_function = coalition_function
        self.coalition_function.to(self.device)

        self.nx_graph = torch_geometric.utils.to_networkx(graph, to_undirected=True)
        self.grand_coalition = list(self.nx_graph.nodes()) # alias: set of players / set of nodes / F
        cc = self.number_connected_components()
        if cc > 1:
            self.log.warning(f"Your graph has {cc} individual components. The worth"
                        " of the grand coalition and the prediction of a GNN can"
                        " differ.")
            pred = self.calculate_prediction()
            worth = self.calculate_worth_of_grand_coalition()
            self.log.warning(f"Prediction={pred:.4f}, Worth={worth:.4f}")

        # if "myerson.fast_restrict" in reversed(sys.modules):
        #     self.fast_restrict_available = True
        # else:
        #     self.fast_restrict_available = False
        # self.set_restrict(self.fast_restrict_available)
    
    # def set_restrict(self, use_fast_restrict: bool) -> None:
    #     """Set wheter to use the fast C++ implementation or the networkX
    #     implementation of `restrict(...)`.

    #     Args:
    #         use_fast_restrict (bool): True or False.
    #     """
    #     if use_fast_restrict:
    #         self._set_variables_for_fast_restrict()
    #         self.log.info("Using fast_restrict() from external C++ library.")
    #         self.restrict = self.fast_restrict
    #     else:
    #         self.log.info("Using python only slow `restrict()` (networkx package).")
    #         self.restrict = super().restrict

    def _forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                 batch: torch.Tensor) -> torch.Tensor:
        """Run the GNN in eval mode without building an autograd graph.

        Returns a detached CPU tensor of shape ``(B, tasks)`` (one row per graph
        in the ``batch`` vector). The model's ``training`` flag is restored.

        Note: toggling ``eval``/``train`` mutates the module's mode in place and is
        not thread-safe. Do not share the model with concurrent train/inference
        threads while an explainer is running.
        """
        model = self.coalition_function
        was_training = model.training
        model.eval()
        try:
            with torch.inference_mode():
                return model(x, edge_index, batch).cpu()
        finally:
            model.train(was_training)

    def _empty_worth(self):
        """Worth assigned to the empty coalition."""
        return 0.

    def _postprocess_worth(self, model_output_row: torch.Tensor) -> float:
        """Convert a single row of the (batched) model output into a worth."""
        return model_output_row.item()

    def calculate_worth_of_single_graph_restricted_coalition(self,
        graph_restricted_coalition: tuple,
        pyg_graph: torch_geometric.data.Data) -> float:
        """Calculate the worth of a graph restricted coalition, i. e. a single
        connected component.

        Args:
            graph_restricted_coalition (tuple): Graph restricted coalition as
                node indices.
            pyg_graph (torch_geometric.data.Data): Graph from which a subgraph
                of the connected components will be extracted according to the
                graph restricted coalition.

        Returns:
            float: Worth, the output of the coalition function for the connected
            subgraph. 
        """
        if not graph_restricted_coalition:
            return self._empty_worth()
        subgraph = self.subgraph_from_coalition(graph_restricted_coalition, pyg_graph)
        out = self._forward(subgraph.x, subgraph.edge_index, self._batch_var(subgraph))
        return self._postprocess_worth(out.squeeze(0))

    def calculate_worth_of_graph_restricted_coalitions(self,
        graph_restricted_coalitions: list,
        batch_size: int = 4096) -> dict:
        """Calculate the worth of every graph restricted coalition and map it to
        its worth.

        The non-empty connected components are evaluated in *batches*: many
        subgraphs are collated into a single disjoint-union PyG graph (one
        ``batch`` vector entry per subgraph) and run through the GNN in one
        forward pass (under ``torch.inference_mode`` + ``eval``).

        Args:
            graph_restricted_coalitions (list): Connected components as tuples of
                node indices.
            batch_size (int, optional): Maximum number of subgraphs per forward
                pass. On GPU throughput rises with batch size up to a few
                thousand; subgraphs are small so 4096 is well within memory.
                Defaults to 4096.

        Returns:
            dict: Dictionary mapping each connected component to its worth.
        """
        self.log.info(f"Calculating worth of graph restricted coalitions.")
        graph_restricted_coalitions_to_worth = {}

        non_empty = []
        for coalition in graph_restricted_coalitions:
            if not coalition:
                graph_restricted_coalitions_to_worth[()] = self._empty_worth()
            else:
                non_empty.append(coalition)

        for start in tqdm(range(0, len(non_empty), batch_size),
                          desc="Calculating worth of graph restricted coalitions",
                          disable=self.disable_tqdm):
            chunk = non_empty[start:start + batch_size]
            x, edge_index, batch = self._batch_data_from_coalitions(chunk)
            out = self._forward(x, edge_index, batch)
            for coalition, worth in zip(chunk, out):
                graph_restricted_coalitions_to_worth[coalition] = \
                    self._postprocess_worth(worth)
        return graph_restricted_coalitions_to_worth

    def _batch_data_from_coalitions(self, coalitions: list):
        """Build one disjoint-union graph for many connected subgraphs in a
        single vectorised pass directly from the parent ``pyg_graph``.

        Returns ``(x, edge_index, batch)`` for ``model(x, edge_index, batch)``,
        equivalent to collating ``subgraph_from_coalition(c)`` for every ``c``
        but with no per-coalition Python iteration over the parent graph: a
        ``(B, n_atoms)`` membership matrix drives all node/edge selection and
        relabelling via ``nonzero`` + fancy indexing.
        """
        pg = self.pyg_graph
        device = pg.x.device
        x_all = pg.x
        edge_index = pg.edge_index
        n_atoms = x_all.shape[0]
        src, dst = edge_index[0], edge_index[1]
        n_batch = len(coalitions)

        # Node membership matrix (B x n_atoms), built with one scatter.
        lengths = torch.tensor([len(c) for c in coalitions], device=device)
        node_batch_in = torch.repeat_interleave(
            torch.arange(n_batch, device=device), lengths)
        all_nodes_in = torch.tensor(
            [node for c in coalitions for node in c],
            dtype=torch.long, device=device)
        member = torch.zeros((n_batch, n_atoms), dtype=torch.bool, device=device)
        member[node_batch_in, all_nodes_in] = True

        # nonzero is row-major -> per-batch ascending node order (== sorted).
        node_batch, node_old = member.nonzero(as_tuple=True)
        n_nodes_total = node_old.shape[0]
        node_new_id = torch.empty((n_batch, n_atoms), dtype=torch.long, device=device)
        node_new_id[node_batch, node_old] = torch.arange(n_nodes_total, device=device)

        # Edges kept iff both endpoints are in the coalition.
        edge_member = member[:, src] & member[:, dst]
        edge_batch, edge_old = edge_member.nonzero(as_tuple=True)
        new_src = node_new_id[edge_batch, src[edge_old]]
        new_dst = node_new_id[edge_batch, dst[edge_old]]
        batched_edge_index = torch.stack([new_src, new_dst])

        x = x_all[node_old]
        return x, batched_edge_index, node_batch

    def calculate_worth_of_grand_coalition(self) -> float:
        """Calculate payoff of the game, i.e. the model prediction. Note that a
        disconnected graph (> 2 molecules) can lead to differeces between 
        the model prediction and this function. 

        Args:
            coalition_function (Callable): The coalition function associating a
                coalition with a payoff.
            nx_graph (nx.classes.graph.Graph): Coalition structure of the game
                as a graph.

        Returns:
            float: Payoff of the game / worth of grand coalition.
        """
        restricted_grand_coalition = self.restrict(self.grand_coalition, self.nx_graph)
        worth = sum([self.calculate_worth_of_single_graph_restricted_coalition(S, self.pyg_graph) \
                    for S in restricted_grand_coalition])
        return worth 

    def calculate_prediction(self) -> float:
        """Calculate the prediction of the GNN for the investigated graph. When 
        the graph is disconnected this prediction may differ from the worth 
        of the grand coalition.

        Returns:
            float: Prediction.
        """
        return self._forward(self.pyg_graph.x, self.pyg_graph.edge_index,
                             self._batch_var(self.pyg_graph)).item()
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

    # def fast_restrict(self, coalition: tuple, nx_graph: nx.classes.graph.Graph) -> list[tuple]:
    #     """Restricts a graph through a (sub)set of nodes / players. Generate a
    #     list of graph restricted coalitions, i. e. a list of node indices of
    #     connected nodes in the subgraph. Uses python wrapped C++ code for 
    #     efficiency.

    #     Args:
    #         coalition (tuple): Nodes that remain in the graph.
    #         nx_graph (nx.classes.graph.Graph): Graph from which to generate
    #             subgraphs.

    #     Returns:
    #         list[tuple]: Graph restricted coalitions as tuples of node indices.
    #     """
    #     remove_nodes = set(nx_graph.nodes)-set(coalition)
    #     if remove_nodes == set(nx_graph.nodes):
    #         return [()] # empty_graph 
    #     component_map = cpp_graph_divide.get_connected_components(self.num_nodes, # type: ignore
    #                                                 self.num_edges,
    #                                                 self.edge_from_ptr,
    #                                                 self.edge_to_ptr,
    #                                                 list(remove_nodes))
    #     connected_subgraph_nodes = {}
    #     seen_component = []
    #     for node, component in enumerate(component_map): 
    #         if (component not in seen_component) and (node not in remove_nodes):
    #             connected_subgraph_nodes.update({component: [node]})
    #             seen_component.append(component)
    #         else:
    #             if node not in remove_nodes:
    #                 connected_subgraph_nodes.update({component: connected_subgraph_nodes[component]+[node]})
    #     return [tuple(connected_subgraph_nodes[key]) for key in connected_subgraph_nodes.keys()]

    def subgraph_from_coalition(self, graph_restricted_coalition: tuple, 
                                pyg_graph: torch_geometric.data.Data) -> torch_geometric.data.Data:
        """Generates a subgraph from a graph restricted coalition (a subset of
        nodes / players) and a graph.

        Args:
            nodes (tuple): Nodes which form the subgraph.
            pyg_graph (torch_geometric.data.Data): Subgraph induced in this
                graph by the subset of nodes.

        Returns:
            torch_geometric.data.Data: The new subgraph.
        """
        # unsorted nodes can result in the wrong edges
        nodes = sorted(graph_restricted_coalition)
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

    # def _set_variables_for_fast_restrict(self) -> None:
    #     """Set class variables for "fast_restrict" function (pointers passed to C++)
    #     """
    #     self.num_nodes = self.pyg_graph.num_nodes
    #     self.edge_from = self.pyg_graph.edge_index.cpu().numpy()[0]
    #     self.edge_to = self.pyg_graph.edge_index.cpu().numpy()[1]
    #     self.edge_from_ptr = self.edge_from.__array_interface__['data'][0]
    #     self.edge_to_ptr = self.edge_to.__array_interface__['data'][0]
    #     self.num_edges = len(self.edge_from)

class MyersonSamplingExplainer(MyersonSampler, MyersonExplainer):
    """A class explaining GNN predictions with approximated Myerson values.

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
        self.log = logging.getLogger("MyersonSamplingExplainer")

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
        cc = self.number_connected_components()
        if cc > 1:
            self.log.warning(f"Your graph has {cc} individual components. The worth"
                        " of the grand coalition and the prediction of a GNN can"
                        " differ.")
            pred = self.calculate_prediction()
            worth = self.calculate_worth_of_grand_coalition()
            self.log.warning(f"Prediction={pred:.4f}, Worth={worth:.4f}")

        # if "myerson.cpp_graph_divide" in sys.modules:
        #     self.fast_restrict_available = True
        # else:
        #     self.fast_restrict_available = False
        # self.set_restrict(self.fast_restrict_available)

class MyersonClassExplainer(MyersonExplainer):
    r"""Explains the prediction of a graph neural network (GNN) classifier with
        Myerson values.  The GNN is treated as the coalition function of a game
        and its prediction as the payoff of the game. The Myerson values show
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
        self.log = logging.getLogger("MyersonClassExplainer")

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.log.info(f"using device {self.device}")
        self.pyg_graph = graph.to(self.device)
        self.coalition_function = coalition_function
        self.coalition_function.to(self.device)

        self.nx_graph = torch_geometric.utils.to_networkx(graph, to_undirected=True)
        self.grand_coalition = list(self.nx_graph.nodes()) # alias: set of players / set of nodes / F
        self.pred = self.calculate_prediction()
        cc = self.number_connected_components()
        if cc > 1:
            self.log.warning(f"Your graph has {cc} individual components. The worth"
                        " of the grand coalition and the prediction of a GNN can"
                        " differ.")
            pred = self.calculate_prediction()
            worth = self.calculate_worth_of_grand_coalition()
            self.log.warning(f"Prediction={pred}, Worth={worth}")
    
    def calculate_worth_of_single_graph_restricted_coalition(self, 
        graph_restricted_coalition: tuple,
        pyg_graph: torch_geometric.data.Data) -> torch.tensor:
        """Calculate the worth of a graph restricted coalition, i. e. a single
        connected component.

        Args:
            graph_restricted_coalition (tuple): Graph restricted coalition as
                node indices.
            pyg_graph (torch_geometric.data.Data): Graph from which a subgraph
                of the connected components will be extracted according to the
                graph restricted coalition.

        Returns:
            tensor: Worth, the output of the coalition function for the connected
            subgraph. 
        """
        if not graph_restricted_coalition:
            return self._empty_worth()
        subgraph = self.subgraph_from_coalition(graph_restricted_coalition, pyg_graph)
        out = self._forward(subgraph.x, subgraph.edge_index, self._batch_var(subgraph))
        return self._postprocess_worth(out.squeeze(0))

    def _empty_worth(self) -> torch.Tensor:
        """Worth assigned to the empty coalition (zero vector over tasks)."""
        return torch.zeros(self.pred.shape)

    def _postprocess_worth(self, model_output_row: torch.Tensor) -> torch.Tensor:
        """Keep the full per-task output vector for the (multi-output) classifier."""
        return model_output_row.clone()

    def calculate_prediction(self) -> torch.tensor:
        """Calculate the prediction of the GNN for the investigated graph. When 
        the graph is disconnected this prediction may differ from the worth 
        of the grand coalition.

        Returns:
            float: Prediction.
        """
        return self._forward(self.pyg_graph.x, self.pyg_graph.edge_index,
                             self._batch_var(self.pyg_graph)).squeeze(0)


class MyersonSamplingClassExplainer(MyersonSamplingExplainer, MyersonClassExplainer):
    """A class explaining a GNNs classifier predictions with approximated Myerson values.

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
        self.log = logging.getLogger("MyersonSamplingClassExplainer")

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
        cc = self.number_connected_components()
        if cc > 1:
            self.log.warning(f"Your graph has {cc} individual components. The worth"
                        " of the grand coalition and the prediction of a GNN can"
                        " differ.")
            pred = self.calculate_prediction()
            worth = self.calculate_worth_of_grand_coalition()
            self.log.warning(f"Prediction={pred}, Worth={worth}")

    def map_coalition_to_worth(self, coalitions: list[tuple], 
                       coalitions_to_graph_restricted_coalitions: dict,
                       graph_restricted_coalitions_to_worth: dict) -> dict:
        """Map every coalition to its worth.

        Args:
            coalitions (list): List of all coalitions (2^{num_nodes}). 
            coalitions_to_graph_restricted_coalitions (dict): Dictionary mapping
                the coalitions to the corresponding graph restricted coalitions.
            graph_restricted_coalitions_to_worth (dict): Dictionary mapping the 
                graph restricted coalitions to their worth.

        Returns:
            dict: Dictionary mapping each coalition to its worth.
        """
        self.log.info(f"Mapping coalitions to worth.")
        coalition_to_worth = {}
        for coalition in tqdm(coalitions, desc="Mapping coalitions to worth", disable=self.disable_tqdm):
            worth = torch.zeros(self.pred.shape)
            for graph_restricted_coalition in coalitions_to_graph_restricted_coalitions[coalition]:
                worth += graph_restricted_coalitions_to_worth[graph_restricted_coalition]
            coalition_to_worth.update({coalition: worth})
        return coalition_to_worth
    
    def sample_all_myerson_values(self) -> np.ndarray:
        """Use Monte Carlo sampling to approximate the Myerson values for every
        node / player in the graph.

        Returns:
            np.ndarray: Sampled Myerson values.
        """
        self.sample_all_mappings()
        self.log.info(f"Calculating sampled Myerson values.")
        # Per-task worth vectors keyed by integer bitmask (see base class).
        worth = {mask: np.asarray(t).squeeze() for mask, t in self._worth_by_mask.items()}
        node_bits = self._node_bits
        random_node_bit = self._random_node_bit
        n = len(node_bits)
        n_tasks = self.pred.shape[0]

        my_values = np.zeros((n, n_tasks), dtype=float)
        for mask in tqdm(self._base_masks, disable=self.disable_tqdm,
                         desc="Calculate sampled Myerson values"):
            for j in range(n):
                bit = node_bits[j]
                without = ((mask ^ bit) | random_node_bit) if (mask & bit) else mask
                my_values[j] += worth[without | bit] - worth[without]

        my_values = my_values / self.number_of_samples
        log_string = "".join([f"\t{node}: {val}\n" for node, val in zip(self.grand_coalition, my_values)])
        self.log.info(f"Sampled Myerson Values:\n{log_string}")
        return my_values

def explain(graph: torch_geometric.data.Data,
            model: torch.nn.Module,
            sample_if_more_nodes_than: int=20,
            verbose: bool=False) -> dict:
    """A function to quickly get started with explaining GNN predictions using Myerson values.

    Args:
        graph (torch_geometric.data.Data): The graph.
        model (torch.nn.Module): The graph neural network.
        sample_if_more_nodes_than (int, optional): Barrier for when to start
            sampling instead of exact calculations. Defaults to 20.
        verbose (bool, optional): Whether to log information to the output and
            show progress bars. Defaults to False.

    Returns:
        dict: The (sampled) Myerson values.
    """

    if verbose:
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s - %(levelname)s] %(message)s', force=True)
        disable_tqdm=False
    else:
        disable_tqdm=True

    node_count = graph.x.size()[0]
    if node_count > sample_if_more_nodes_than:
        logging.info("Sampling Myerson values.")
        sampler = MyersonSamplingExplainer(graph, model, disable_tqdm=disable_tqdm)
        return sampler.sample_all_myerson_values()
    else:
        logging.info("Calculating exact Myerson values.")
        explainer = MyersonExplainer(graph, model, disable_tqdm=disable_tqdm)
        return explainer.calculate_all_myerson_values()
