import math
import networkx as nx
import numpy as np
from itertools import combinations, chain

from typing import Callable

from tqdm import tqdm
import logging

class MyersonBudgetExceeded(Exception):
    """Raised when an exact computation exceeds its connected-subgraph budget.

    Signals a caller (e.g. an ``explain`` helper) that the graph is too large
    for exact Myerson values within the allotted budget and that Monte-Carlo
    sampling should be used instead.
    """


class MyersonCalculator:
    r"""Calculates the exact Myerson values.
        For a game described by a coalition function :math:`v` the Myerson values
        attribute the individual players contribution to the payoff of the game. For
        a complete graph (every node connected to every other node) the Myerson
        value is equal to the Shapley value (:math:`S`: coalition of players,
        :math:`N`: grand coalition of all players):

        .. math::

            \text{Sh}_i\,({v}) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! \: (|N| - |S| - 1)!}{|N|!}\big( {v}\,(S \cup \{i\}) - {v}\,(S) \big)

        Else, the additional gain players can obtain through coalition is
        restricted only to players which are connected by edges in the graph.

    Args:
        graph (nx.classes.graph.Graph): The coalition structure of the game.
        coalition_function (Callable): The coalition for which to calculate
            the payoff of the game. Expects a coalition (tuple of node
            indices) and a graph which contains additional information on
            the players to decide on the payoff.

        disable_tqdm (bool, optional): Disables progress bar. Defaults to
            True.
    """
    #: Optional exact-enumeration budget. If set, exact calculation raises
    #: :class:`MyersonBudgetExceeded` once more than this many connected induced
    #: subgraphs are generated.
    _connected_enum_max_subgraphs: int | None = None  # budget -> MyersonBudgetExceeded

    def __init__(self,
                 graph: nx.classes.graph.Graph,
                 coalition_function: Callable,
                 disable_tqdm: bool=True) -> None:
        """Instantiate the class.
        """

        self.disable_tqdm = disable_tqdm
        self.log = logging.getLogger("MyersonCalculator")
        self.nx_graph = graph
        self.grand_coalition = list(graph.nodes()) # alias: set of players / set of nodes / F
        self.coalition_function = coalition_function

    def calculate_coalitions(self, grand_coalition: list) -> list:
        r"""Calculate all possible coalitions for a set of players / all nodes in
        a graph.

        Args:
            grand_coalition (list): Set of players / grand coalition / atoms in
                graph as a list of tuples.

        Returns:
            list: All :math:`2^N` coalitions.
        """
        self.log.info("Calculating number of coalitions.")
        coalitions = [combinations(grand_coalition, len(grand_coalition)-x) for x in \
                      tqdm(range(len(grand_coalition)), desc="Calculate coalitions", disable=self.disable_tqdm)]
        coalitions = list(chain.from_iterable(coalitions)) # chaining removes empty set
        coalitions.append(())
        self.log.info(f"Number of coalitions: {len(coalitions)}")
        return coalitions

    def calculate_graph_restricted_coalitions(self, coalitions: list,
                                  nx_graph: nx.classes.graph.Graph) -> tuple[set, dict]:
        """Calculate the graph restricted coalitions for each coalition. The
        graph restricted coalitions are tuples of nodes which are connected.

        Args:
            coalitions (list): All coalitions.
            nx_graph (nx.classes.graph.Graph): NetworkX Graph for which to
            calculate the Myerson values.

        Returns:
            tuple[set, dict]: Set of all possible graph restricted coalitions as
                a tuple of nodes, dictionary mapping each coalition to its
                graph restricted coalitions.
        """
        self.log.info("Calculating number of graph restricted coalitions.")
        graph_restricted_coalitions = [self.restrict(coalition, nx_graph) \
                                       for coalition in tqdm(coalitions,
                                           desc="Calculate graph restricted coalitions",
                                           disable=self.disable_tqdm)]
        coalitions_to_graph_restricted_coalitions = dict(zip(coalitions, graph_restricted_coalitions, strict=True))
        graph_restricted_coalitions = list(chain.from_iterable(graph_restricted_coalitions))
        self.log.info(f"Removing duplicates from {len(graph_restricted_coalitions)} graph restricted coalitions.")
        graph_restricted_coalitions = set(x for x in tqdm(graph_restricted_coalitions, desc="Remove duplicates", disable=self.disable_tqdm))
        self.log.info(f"Number of graph restricted coalitions: {len(graph_restricted_coalitions)}")
        return graph_restricted_coalitions, coalitions_to_graph_restricted_coalitions

    def calculate_worth_of_single_graph_restricted_coalition(self,
        graph_restricted_coalition: tuple,
        nx_graph: nx.classes.graph.Graph) -> float:
        """Calculate the worth of a graph restricted coalition, i. e. a single
        connected component.

        Args:
            graph_restricted_coalition (tuple): Graph restricted coalition as
                node indices.
            nx_graph (nx.classes.graph.Graph): Additional information for the
                coalition function, i.e. the entire graph with node parameters.
                The result of the coalition function should depend only on the
                coalition, however the node parameters might contain necessary
                information.

        Returns:
            float: Worth, the output of the coalition function for the connected
            subgraph.
        """
        return self.coalition_function(graph_restricted_coalition, nx_graph)

    def calculate_worth_of_graph_restricted_coalitions(self,
        graph_restricted_coalitions: set) -> dict:
        """Calculate the worth of every graph restricted coalition and map it to
        its worth.

        Args:
            graph_restricted_coalitions (set): Set of connected components as
                tuples of node indices.

        Returns:
            dict: Dictionary mapping each connected component to its worth.
        """
        self.log.info("Calculating worth of graph restricted coalitions.")
        graph_restricted_coalitions_to_worth = {}
        for coalition in tqdm(graph_restricted_coalitions,
                              desc="Calculating worth of graph restricted coalitions",
                              disable=self.disable_tqdm):
            worth = self.calculate_worth_of_single_graph_restricted_coalition(coalition,
                                                                              self.nx_graph)
            graph_restricted_coalitions_to_worth[coalition] = worth
        return graph_restricted_coalitions_to_worth

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
        self.log.info("Mapping coalitions to worth.")
        coalition_to_worth = {}
        for coalition in tqdm(coalitions, desc="Mapping coalitions to worth", disable=self.disable_tqdm):
            worth = 0.
            for graph_restricted_coalition in coalitions_to_graph_restricted_coalitions[coalition]:
                worth += graph_restricted_coalitions_to_worth[graph_restricted_coalition]
            coalition_to_worth[coalition] = worth
        return coalition_to_worth

    def calculate_worth_of_grand_coalition(self, nx_graph: nx.classes.graph.Graph) -> float:
        """Calculate payoff of the game, i.e. the payoff of all players / the
        grand coalition.

        Args:
            nx_graph (nx.classes.graph.Graph): Coalition structure of the game
                as a graph.

        Returns:
            float: Payoff of the game / worth of grand coalition.
        """
        restricted_grand_coalition = self.restrict(self.grand_coalition, nx_graph)
        worth = sum([self.calculate_worth_of_single_graph_restricted_coalition(S, nx_graph) \
                     for S in restricted_grand_coalition])
        return worth

    def _get_adjacency_masks(self, nx_graph: nx.classes.graph.Graph):
        """Build (and cache) a bitmask adjacency representation of ``nx_graph``.

        Each node is assigned an integer index ``0..N-1`` (in node iteration
        order). ``neighbor_masks[i]`` is an integer whose set bits are the
        indices of the neighbours of node ``i``. This lets ``restrict`` compute
        connected components with pure-integer bit operations.

        The cache is keyed on the identity of the graph object, so it is rebuilt
        automatically if a different graph is passed.

        Returns:
            tuple: ``(index_to_label, label_to_index, neighbor_masks)``.

            - ``index_to_label`` (list): maps integer index ``i`` to its node
              label, i.e. ``index_to_label[i]`` is the ``nx_graph`` node at
              index ``i``. Labels are whatever the graph uses (any hashable;
              usually ``int``).
            - ``label_to_index`` (dict): the inverse mapping, from node label to
              integer index.
            - ``neighbor_masks`` (list[int]): one bitmask per index;
              ``neighbor_masks[i]`` has bit ``j`` set iff nodes ``i`` and ``j``
              are adjacent. Self-loops are ignored.
        """
        if getattr(self, "_adj_graph_id", None) == id(nx_graph):
            return (self._adj_index_to_label,
                    self._adj_label_to_index,
                    self._adj_neighbor_masks)

        label_to_index = {label: i for i, label in enumerate(nx_graph.nodes())}
        index_to_label = list(label_to_index.keys())
        neighbor_masks = [0] * len(index_to_label)
        for u, v in nx_graph.edges():
            iu = label_to_index[u]
            iv = label_to_index[v]
            if iu == iv:
                continue  # ignore self-loops
            neighbor_masks[iu] |= (1 << iv)
            neighbor_masks[iv] |= (1 << iu)

        self._adj_graph_id = id(nx_graph)
        self._adj_label_to_index = label_to_index
        self._adj_index_to_label = index_to_label
        self._adj_neighbor_masks = neighbor_masks
        return index_to_label, label_to_index, neighbor_masks

    @staticmethod
    def _connected_component_masks(node_mask: int,
                                   neighbor_masks: list[int]) -> list[int]:
        """Split a bitmask of nodes into its connected-component bitmasks.

        Grows components with integer-bitmask BFS over ``neighbor_masks`` (from
        :meth:`_get_adjacency_masks`). Edges leaving ``node_mask`` are ignored,
        so each returned integer is a connected component of the subgraph
        induced by ``node_mask``. Components are yielded lowest-node-first and
        the empty mask yields ``[]``.

        Args:
            node_mask (int): Bitmask of the nodes to partition. Bit ``i`` set
                means index ``i`` is present.
            neighbor_masks (list[int]): Adjacency bitmasks; ``neighbor_masks[i]``
                has bit ``j`` set iff indices ``i`` and ``j`` are adjacent.

        Returns:
            list[int]: One bitmask per connected component.
        """
        components = []
        remaining = node_mask
        while remaining:
            # Grow a component starting from the lowest remaining node.
            frontier = remaining & (-remaining)
            component = 0
            while frontier:
                component |= frontier
                neighbours = 0
                f = frontier
                while f:
                    low = f & (-f)
                    neighbours |= neighbor_masks[low.bit_length() - 1]
                    f ^= low
                # Only follow edges that stay inside node_mask.
                frontier = neighbours & remaining & ~component
            remaining &= ~component
            components.append(component)
        return components

    @staticmethod
    def _mask_to_labels(node_mask: int, index_to_label: list) -> tuple:
        """Expand a node bitmask into a tuple of node labels.

        Nodes come out in ascending index order (deterministic), so the result
        is a stable dictionary key.

        Args:
            node_mask (int): Bitmask of the nodes; bit ``i`` set means index
                ``i`` is present.
            index_to_label (list): Maps integer index ``i`` to its node label
                (see :meth:`_get_adjacency_masks`).

        Returns:
            tuple: Node labels of the set bits, ascending by index.
        """
        labels = []
        m = node_mask
        while m:
            low = m & (-m)
            labels.append(index_to_label[low.bit_length() - 1])
            m ^= low
        return tuple(labels)

    def number_connected_components(self, nx_graph: nx.classes.graph.Graph | None = None) -> int:
        """Count the connected components of ``nx_graph`` (default: ``self.nx_graph``).

        Uses the cached integer-bitmask adjacency (see :meth:`_get_adjacency_masks`)
        rather than a NetworkX graph algorithm, so it has no extra dependency on
        NetworkX traversal routines.

        Args:
            nx_graph (nx.classes.graph.Graph, optional): Graph to inspect.
                Defaults to ``self.nx_graph``.

        Returns:
            int: Number of connected components.
        """
        if nx_graph is None:
            nx_graph = self.nx_graph
        index_to_label, _, _ = self._get_adjacency_masks(nx_graph)
        grand_coalition = tuple(index_to_label)
        if not grand_coalition:
            return 0
        return len(self.restrict(grand_coalition, nx_graph))

    def restrict(self, coalition: tuple, nx_graph: nx.classes.graph.Graph) -> list[tuple]:
        """Restricts a graph through a (sub)set of nodes / players. Generate a
        list of graph restricted coalitions, i. e. a list of node indices of
        connected nodes in the subgraph.

        Connected components are computed with integer-bitmask BFS over a cached
        adjacency representation (see :meth:`_get_adjacency_masks`), which is
        substantially faster than constructing a NetworkX subgraph view per
        coalition. Each returned tuple lists its nodes in ascending index order,
        which is deterministic so the tuples are stable dictionary keys.

        Args:
            coalition (tuple): Nodes that remain in the graph.
            nx_graph (nx.classes.graph.Graph): Graph from which to generate
                subgraphs.

        Returns:
            list[tuple]: Graph restricted coalitions as tuples of node indices.
        """
        if not coalition:
            return [()] # empty_graph

        index_to_label, label_to_index, neighbor_masks = \
            self._get_adjacency_masks(nx_graph)

        # Bitmask of the nodes that remain in the graph.
        node_mask = 0
        for label in coalition:
            node_mask |= (1 << label_to_index[label])

        return [self._mask_to_labels(component, index_to_label)
                for component in self._connected_component_masks(
                    node_mask, neighbor_masks)]
    def subgraph_from_coalition(self, graph_restricted_coalition: tuple,
                               nx_graph: nx.classes.graph.Graph) -> nx.classes.graph.Graph:
        """Generates a subgraph from a graph restricted coalition (a subset of
        nodes / players) and a graph.

        Args:
            graph_restricted_coalition (tuple): Nodes / players which form the
                subgraph.
            nx_graph (nx.classes.graph.Graph): Subgraph induced in this graph by
                the nodes_set.

        Returns:
            nx.classes.graph.Graph: The new subgraph.
        """
        if not graph_restricted_coalition:
            return nx.Graph()
        else:
            return nx_graph.subgraph(graph_restricted_coalition)

    def connected_subgraph_count_lower_bound(
            self,
            nx_graph: nx.classes.graph.Graph | None = None,
            stop_above: int | None = None) -> int:
        """Lower-bound the number of connected induced subgraphs cheaply.

        The count is computed exactly on a spanning forest of ``nx_graph``. Every
        connected induced subgraph of that forest is also connected in the
        original graph, so this is a safe lower bound for the exact enumeration
        cost. For trees / forests (common molecule backbones without rings) the
        lower bound is the exact count. If ``stop_above`` is given, intermediate
        values are capped and the method returns as soon as the lower bound is
        known to exceed that threshold.

        This is useful as a preflight for budgeted exact explanation: if even a
        spanning forest has more connected subgraphs than the budget, the full
        graph certainly does too, so callers can switch to sampling without
        starting the connected-subgraph enumerator.

        Args:
            nx_graph (nx.classes.graph.Graph, optional): Graph to inspect.
                Defaults to ``self.nx_graph``.
            stop_above (int | None, optional): Early-exit threshold. If supplied,
                the returned value may be capped at ``stop_above + 1`` once the
                lower bound is known to be above the threshold.

        Returns:
            int: A lower bound on the number of non-empty connected induced
            subgraphs.
        """
        if nx_graph is None:
            nx_graph = self.nx_graph
        index_to_label, _, neighbor_masks = self._get_adjacency_masks(nx_graph)
        n = len(index_to_label)
        if n == 0:
            return 0

        cap = stop_above + 1 if stop_above is not None else None
        full = (1 << n) - 1
        unvisited = full
        children = [[] for _ in range(n)]
        postorder: list[int] = []

        def _capped(value: int) -> int:
            if cap is not None and value > cap:
                return cap
            return value

        # DFS to find spanning trees
        while unvisited:
            # Pick a high-degree root for this component. The bound is valid for
            # any spanning tree; high-degree roots tend to give a tighter lower
            # bound on branched/dense molecular graphs.
            root = max(
                (i for i in range(n) if unvisited & (1 << i)),
                key=lambda i: (neighbor_masks[i].bit_count(), -i),
            )
            unvisited &= ~(1 << root)
            stack = [root]
            while stack:
                u = stack.pop()
                postorder.append(u)
                nbrs = neighbor_masks[u] & unvisited
                while nbrs:
                    bit = nbrs & -nbrs
                    nbrs ^= bit
                    v = bit.bit_length() - 1
                    unvisited &= ~bit
                    children[u].append(v)
                    stack.append(v)

        # count connected vertex subsets of the spanning forest, basically all subpaths in the tree
        # which is a lower bound to all induced subgraphs in the real graph.
        containing_root = [0] * n
        total = 0
        for u in reversed(postorder):
            count = 1                          # the empty choice for each child branch
            for v in children[u]:
                count = _capped(count * (1 + containing_root[v]))
            containing_root[u] = count
            total = _capped(total + count)
            if cap is not None and total > stop_above:
                return total
        return total

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

    def calculate_single_myerson_value(self, node: int, grand_coalition: tuple,
                                  coalitions: list[tuple], coalition_to_worth: dict) -> float:
        """Calculate a single Myerson value.

        Args:
            node (int): Node index for which to calculate the Myerson value.
            grand_coalition (tuple): Set of all players.
            coalitions (list[tuple]): List of all coalitions.
            coalition_to_worth (dict): Mapping of every coalition to its worth.

        Returns:
            float: Myerson value.
        """
        my = 0
        size_grand_coalition = len(grand_coalition)
        prefactors = self._precompute_prefactors(size_grand_coalition)
        for coalition in coalitions:
            if node in coalition:
                continue
            size_coalition = len(coalition)
            worth_of_coalition = coalition_to_worth[coalition]
            worth_of_coalition_with_node = coalition_to_worth[tuple(sorted(coalition+(node,)))]
            my += prefactors[size_coalition] * (worth_of_coalition_with_node - worth_of_coalition)
        return my

    def _enumerate_connected_subgraphs(
            self, max_subgraphs: int | None = None) -> tuple[list, list, list]:
        """Enumerate every connected induced subgraph of ``self.nx_graph``
        exactly once (Skibski et al. 2014, Algorithm 1 - an ESU-style DFS
        enumeration in the sense of Wernicke 2006). This is an iterative,
        bitmask variant: ``below``/``seen`` play the role of the paper's
        forbidden set ``X`` and the per-root loop enforces the min-index-root
        uniqueness invariant. Vertices are pre-ordered by degree descending
        (paper Alg. 1, line 2) to shrink later roots' frontiers. The resulting
        ``(S, N(S))`` pairs feed the Skibski et al. 2014 neighbour-set weighting
        in :meth:`_calculate_all_myerson_values_connected_enum`.

        This never touches the 2^N coalition lattice: it enumerates only the |C|
        *connected induced subgraphs* and uses the closed-form neighbor-set
        weighting. Time O(|C| * |E|), memory O(|C|). For sparse graphs (molecules)
        |C| << 2^N.

        Skibski, O., Michalak, T.P., Rahwan, T. and Wooldridge, M., 2014.
        Algorithms for the Shapley and Myerson values in graph-restricted games.
        In Proceedings of the 2014 international conference on Autonomous agents and
        multi-agent systems (pp. 197-204).

        Wernicke, S., 2006.
        Efficient detection of network motifs.
        IEEE/ACM transactions on computational biology and bioinformatics,
        3(4), pp.347-359.

        Args:
            max_subgraphs (int, optional): If given, raise
                :class:`MyersonBudgetExceeded` as soon as more than this many
                connected subgraphs have been generated. Lets a caller bail out
                to sampling on graphs whose exact cost (``|C|`` model forwards)
                is too high.

        Returns:
            tuple(list[int], list[tuple], list[int]): Three parallel lists,
            one entry per connected subgraph ``S`` (index ``i`` refers to the
            same ``S`` in all three):

                * ``sub_masks``: ``S`` as an integer bitmask; bit ``k`` set means
                  the vertex at adjacency index ``k`` is in ``S``.
                * ``sub_tuples``: the same ``S`` as a tuple of node labels
                  (``self.nx_graph`` node ids), passed to the worth oracle.
                * ``neighbour_masks_of_sub``: bitmask of ``N(S)``, the vertices
                  adjacent to ``S`` but not in it (``nbrs(S) \\ S``), used for the
                  Skibski neighbour-set weighting.
        """
        index_to_label, _, neighbor_masks = \
            self._get_adjacency_masks(self.nx_graph)
        n = len(index_to_label)

        if max_subgraphs is not None:
            lower_bound = self.connected_subgraph_count_lower_bound(
                stop_above=max_subgraphs)
            if lower_bound > max_subgraphs:
                raise MyersonBudgetExceeded(
                    f"At least {lower_bound} connected subgraphs; "
                    f"budget is {max_subgraphs}, so exact enumeration was "
                    "skipped before starting (use sampling instead).")

        # Traverse in a degree-descending vertex order (Skibski et al. 2014,
        # Alg. 1, line 2). Rooting at (and forbidding) high-degree vertices first
        # shrinks the frontier explored by later, lower-degree roots. Correctness
        # only needs a consistent total order, so we relabel to "rank" space
        # (rank 0 = highest degree) for the traversal and map back afterwards.
        rank_to_index = sorted(
            range(n), key=lambda i: (-bin(neighbor_masks[i]).count("1"), i))
        index_to_rank = [0] * n
        for rank, idx in enumerate(rank_to_index):
            index_to_rank[idx] = rank
        rank_neighbor_masks = [0] * n
        for rank, idx in enumerate(rank_to_index):
            m = neighbor_masks[idx]
            remapped = 0
            while m:
                low = m & (-m)
                remapped |= 1 << index_to_rank[low.bit_length() - 1]
                m ^= low
            rank_neighbor_masks[rank] = remapped

        sub_masks_rank: list[int] = []
        for root in range(n):
            below = (1 << root) - 1                 # all lower-rank vertices
            init_ext = rank_neighbor_masks[root] & ~below
            # `seen` = vertices that may never (re-)enter the frontier: the root,
            # all lower vertices, and everything already queued in the frontier.
            seen0 = below | (1 << root) | init_ext
            # Stack items are (sub, ext, seen): the current subgraph, its
            # extension frontier (candidates to add), and the forbidden set.
            stack = [(1 << root, init_ext, seen0)]
            while stack:
                sub, ext, seen = stack.pop()
                sub_masks_rank.append(sub)         # sub is a finished subgraph
                if max_subgraphs is not None \
                        and len(sub_masks_rank) > max_subgraphs:
                    raise MyersonBudgetExceeded(
                        f"More than {max_subgraphs} connected subgraphs; "
                        "exact enumeration aborted (use sampling instead).")
                e = ext                            # remaining candidates to branch on
                while e:
                    w = e & (-e)                   # next candidate vertex (bit)
                    e ^= w                          # consume w at this level
                    wi = w.bit_length() - 1        # its vertex (rank) index
                    new_nbrs = rank_neighbor_masks[wi] & ~seen  # exclusive neighbors
                    # Child keeps the remaining siblings (`e`) plus w's new
                    # exclusive neighbors; `seen` accumulates only down a branch.
                    stack.append((sub | w, e | new_nbrs, seen | new_nbrs))

        # Map each rank-space subgraph back to original-index space: `sub_masks`
        # and `neighbour_masks_of_sub` are bit-indexed by original adjacency
        # index (the space the caller's Myerson accumulator uses).
        sub_masks: list[int] = []
        sub_tuples: list[tuple] = []
        neighbour_masks_of_sub: list[int] = []
        for rm in sub_masks_rank:                  # rm: rank-space subgraph mask
            orig_mask = 0                          # same subgraph in original space
            labels = []
            m = rm
            while m:                               # walk the vertices of the subgraph
                low = m & (-m)
                idx = rank_to_index[low.bit_length() - 1]  # rank -> original index
                orig_mask |= 1 << idx
                labels.append(index_to_label[idx])
                m ^= low
            sub_masks.append(orig_mask)
            sub_tuples.append(tuple(labels))
            nbr = 0                                # accumulate N(S) in original space
            c = orig_mask
            while c:
                low = c & (-c)
                nbr |= neighbor_masks[low.bit_length() - 1]
                c ^= low
            neighbour_masks_of_sub.append(nbr & ~orig_mask)  # N(S) = nbrs \ S
        return sub_masks, sub_tuples, neighbour_masks_of_sub


    @staticmethod
    def _coerce_worth(worth):
        """Normalise a worth into ``(value, is_scalar)``.

        Scalar worths (pure game / single-output GNN) stay Python floats; tensor
        / array worths (multi-output classifier) become a flat ``float64`` array
        so the same accumulation loop serves both. Torch CPU tensors implement
        ``__array__`` so ``np.asarray`` handles them.
        """
        if isinstance(worth, (int, float, np.integer, np.floating)):
            return float(worth), True
        return np.asarray(worth, dtype=np.float64).reshape(-1), False

    def _calculate_all_myerson_values_connected_enum(
            self, max_subgraphs: int | None = None) -> np.ndarray:
        """Exact Myerson values via connected-subgraph enumeration (Skibski et
        al. 2014, Algorithm 4).

        Skibski, O., Michalak, T.P., Rahwan, T. and Wooldridge, M., 2014, May.
        Algorithms for the Shapley and Myerson values in graph-restricted games.
        In Proceedings of the 2014 international conference on Autonomous agents and
        multi-agent systems (pp. 197-204).

        For every connected induced subgraph ``S`` with full-graph neighbour set
        ``N(S)`` and worth ``v(S)``::

            for u in S      : MV[u] += (|S|-1)! |N(S)|!   / (|S|+|N(S)|)! * v(S)
            for u in N(S)   : MV[u] -= |S|!   (|N(S)|-1)! / (|S|+|N(S)|)! * v(S)

        The worth oracle ``v`` is evaluated exactly once per connected subgraph.
        Handles both scalar worths and tensor / multi-output worths (the latter
        returns a ``(n_nodes, n_tasks)`` array, like the class sampler).

        Args:
            max_subgraphs (int, optional): Budget forwarded to
                :meth:`_enumerate_connected_subgraphs`; raises
                :class:`MyersonBudgetExceeded` if exceeded.

        Returns:
            np.ndarray: Myerson values in ``grand_coalition`` order, shaped
            ``(n,)`` for scalar worths or ``(n, n_tasks)`` for tensor worths.
        """
        index_to_label, label_to_index, _ = \
            self._get_adjacency_masks(self.nx_graph)
        n = len(index_to_label)
        if n == 0:
            return np.array([], dtype=np.float64)

        sub_masks, sub_tuples, neighbour_masks = \
            self._enumerate_connected_subgraphs(max_subgraphs=max_subgraphs)

        # One batched worth evaluation over all connected subgraphs (the GNN
        # forward). Could be chunked to bound memory to O(batch) instead of O(C).
        worth_by_tuple = self.calculate_worth_of_graph_restricted_coalitions(
            sub_tuples)
        self.graph_restricted_coalitions = set(sub_tuples)
        self.graph_restricted_coalitions_to_worth = worth_by_tuple

        # Decide scalar vs tensor (multi-output) accumulation from a sample worth.
        sample = next(iter(worth_by_tuple.values())) if worth_by_tuple else 0.0
        _, scalar = self._coerce_worth(sample)
        if scalar:
            mv = np.zeros(n, dtype=np.float64)
        else:
            n_tasks = self._coerce_worth(sample)[0].shape[0]
            mv = np.zeros((n, n_tasks), dtype=np.float64)

        fact = [math.factorial(k) for k in range(n + 1)]
        # Accumulate Skibski Alg. 4 lines 19-22: one weighted worth per subgraph.
        for mask, tup, nbr in zip(sub_masks, sub_tuples, neighbour_masks, strict=True):
            value, _ = self._coerce_worth(worth_by_tuple[tup])  # v(S)
            if scalar and value == 0.0:
                continue                       # contributes nothing to any node
            s = len(tup)                       # |S|
            t = bin(nbr).count("1")            # |N(S)|
            denom = fact[s + t]                # (|S|+|N(S)|)!
            # Positive weight for members of S (all of S before u, N(S) after u).
            pos = fact[s - 1] * fact[t] / denom
            m = mask                           # walk the bits (vertices) of S
            while m:
                low = m & (-m)                 # lowest set bit
                mv[low.bit_length() - 1] += pos * value
                m ^= low                       # clear it, advance
            if t:
                # Negative weight for neighbours of S (adding u reconnects S).
                neg = fact[s] * fact[t - 1] / denom
                c = nbr                        # walk the bits (vertices) of N(S)
                while c:
                    low = c & (-c)
                    mv[low.bit_length() - 1] -= neg * value
                    c ^= low

        # Reorder from internal vertex-index order into grand_coalition order.
        order = [label_to_index[label] for label in self.grand_coalition]
        return mv[order]

    def calculate_all_myerson_values(self) -> np.ndarray:
        """Calculate the Myerson values for every node / player in the graph.

        The production exact strategy is connected-subgraph enumeration
        (:meth:`_calculate_all_myerson_values_connected_enum`). It evaluates the
        coalition function once per connected induced subgraph and uses the
        Skibski et al. closed-form neighbour-set weighting to avoid the full
        ``2^N`` coalition lattice. A ``_connected_enum_max_subgraphs`` budget
        (if set) makes it raise :class:`MyersonBudgetExceeded` instead of
        grinding through an intractable ``|C|``.

        Returns:
            np.ndarray: Myerson values for each node (shape ``(n,)`` for scalar
            worths, ``(n, n_tasks)`` for tensor / multi-output worths).
        """
        budget = getattr(self, "_connected_enum_max_subgraphs", None)
        my_values = self._calculate_all_myerson_values_connected_enum(
            max_subgraphs=budget)
        log_string = "".join([f"\t{node}: {val}\n" for node, val in zip(self.grand_coalition, my_values, strict=True)])
        self.log.info(f"Myerson Values:\n{log_string}")
        return my_values

    def calculate_all_mappings(self) -> None:
        """Legacy full-lattice pipeline: enumerate all ``2^N`` coalitions, their
        graph restricted coalitions, and the associated worths as class
        attributes. The production exact path is
        :meth:`calculate_all_myerson_values` (connected-subgraph enumeration),
        which never materialises the coalition lattice; this method is kept for
        callers that need the full per-coalition mappings.

            * `self.coalitions` (list[tuple])
            * `self.graph_restricted_coalitions` (set[tuple])
            * `self.coalitions_to_graph_restricted_coalitions` (dict)
            * `self.graph_restricted_coalitions_to_worth` (dict)
            * `self.coalitions_to_worth` (dict)
        """
        self.coalitions = self.calculate_coalitions(self.grand_coalition)

        self.graph_restricted_coalitions, self.coalitions_to_graph_restricted_coalitions \
            = self.calculate_graph_restricted_coalitions(self.coalitions, self.nx_graph)

        self.graph_restricted_coalitions_to_worth \
            = self.calculate_worth_of_graph_restricted_coalitions(self.graph_restricted_coalitions)

        self.coalitions_to_worth \
            = self.map_coalition_to_worth(self.coalitions,
                                          self.coalitions_to_graph_restricted_coalitions,
                                          self.graph_restricted_coalitions_to_worth)

class MyersonSampler(MyersonCalculator):
    r"""A class approximating the Myerson value using Monte Carlo sampling.
        The Myerson values are approximated by randomly sampling from all
        permutations needed to calculate the Shapley value:

        .. math::

            \text{Sh}_i\,({v}) = \frac{1}{|N|!}\; \sum_R \big({v}\,(P_i^R \cup \{i\}) - {v}(P_i^R)\big)

        For efficiencies sake, the sampled permutations are transformed into
        the corresponding coalitions.

    Args:
        graph (nx.classes.graph.Graph): The coalition structure of the game.
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
        """Instantiate the class.
        """
        super().__init__(graph, coalition_function, disable_tqdm)
        self.log = logging.getLogger("MyersonSampler")
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.number_of_samples = number_of_samples

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

    def _sample_base_permutations(self, number_of_samples: int):
        """RNG-driven core of :meth:`sample_permutations`: draw the base
        prefix-coalitions, each excluding the randomly chosen pivot node.

        Returns:
            tuple(int, list[np.ndarray]): the pivot node, and the sampled
            prefix-coalitions (without the pivot node).
        """
        nodes_array = np.array(self.grand_coalition)
        random_node = self.rng.choice(nodes_array)

        # The pivot is fixed for the whole sampling run, so the pool we draw the
        # prefix from ("all nodes except the pivot") is invariant.
        # This consumes no RNG, so the sampled permutations are unchanged.
        nodes_array_without_random_node = np.delete(
            nodes_array, np.where(nodes_array == random_node))

        self.log.info(f"Sampling {number_of_samples} steps.")
        permutations_without_random_node = []
        for i in tqdm(range(number_of_samples),
                      desc="Sample permutations without random node",
                      disable=self.disable_tqdm):
            random_permutation_size: int = self.rng.integers(0, len(nodes_array))
            sampled_permutation_without_random_node = self.rng.choice(nodes_array_without_random_node,
                                                        size=random_permutation_size,
                                                        replace=False)
            self.rng.shuffle(sampled_permutation_without_random_node)
            permutations_without_random_node.append(sampled_permutation_without_random_node)
        return random_node, permutations_without_random_node

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
        random_node, permutations_without_random_node = \
            self._sample_base_permutations(number_of_samples)
        nodes_array = np.array(self.grand_coalition)

        all_sampled_permutations = []
        for permutation in tqdm(permutations_without_random_node,
                         desc="Sample permutations containing random node",
                         disable=self.disable_tqdm):
            for node_idx, node in enumerate(nodes_array):
                swapped = self._replace_in_array(permutation.copy(), node, random_node)
                all_sampled_permutations.append(swapped)
                all_sampled_permutations.append(np.append(swapped, node))
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

    def _build_sampling_worth_by_mask(self) -> dict:
        """Enumerate the sampled connected components, evaluate their worth, and
        build the bitmask-keyed worth table (single-molecule path).
        """
        components = self._enumerate_sampling_components()
        graph_restricted_coalitions_to_worth = \
            self.calculate_worth_of_graph_restricted_coalitions(components)
        return self._finish_sampling_worth_by_mask(
            graph_restricted_coalitions_to_worth)

    def _enumerate_sampling_components(self) -> list:
        """Build the integer-bitmask machinery the estimator's value loop needs,
        and return the connected components whose worth must be evaluated.

        From the sampled base coalitions (``self.permutations_without_random_node``)
        and the pivot (``self.random_node``) this computes:

            * ``self._node_bits`` — ``1 << i`` for each player ``i``.
            * ``self._random_node_bit`` — the pivot's bit.
            * ``self._base_masks`` — each base coalition as a bitmask.

        and the exact set of coalitions the estimator will query. The connected
        components (whose worth — possibly an NN forward — is the expensive part)
        are returned so the caller can evaluate them (single-molecule, or pooled
        across molecules); :meth:`_finish_sampling_worth_by_mask` then assembles
        the worth table.

        Connected components are computed with the §4.1 ``(S, S∪{j})`` pairing:
        the queried coalitions come in with/without-``j`` pairs, so we compute
        ``components(S)`` once (bitmask BFS, memoised) and derive
        ``components(S∪{j})`` by merging the components adjacent to ``j`` (a few
        bitwise ANDs) instead of an independent connected-components pass.
        """
        index_to_label, label_to_index, neighbor_masks = \
            self._get_adjacency_masks(self.nx_graph)
        n = len(self.grand_coalition)
        node_bits = [1 << i for i in range(n)]
        random_node_bit = 1 << label_to_index[self.random_node]

        base_masks = []
        for perm in self.permutations_without_random_node:
            mask = 0
            for label in perm.tolist():
                mask |= 1 << label_to_index[label]
            base_masks.append(mask)

        # components_of[mask] -> list of connected-component bitmasks ([] for the
        # empty coalition). Memoised across the whole sampled query set.
        components_of: dict[int, list[int]] = {}

        # For each base coalition and player j, build the with/without-j pair.
        # components(without) is computed once (memoised); components(with) is
        # derived by the single-vertex merge (§4.1) — no second BFS.
        for mask in base_masks:
            for j in range(n):
                bit = node_bits[j]
                without = ((mask ^ bit) | random_node_bit) if (mask & bit) else mask
                with_node = without | bit

                comps_without = components_of.get(without)
                if comps_without is None:
                    comps_without = self._connected_component_masks(
                        without, neighbor_masks)
                    components_of[without] = comps_without

                if with_node not in components_of:
                    neighbours_of_j = neighbor_masks[j]
                    merged = bit
                    kept = []
                    for c in comps_without:
                        if c & neighbours_of_j:   # component touches j -> merge
                            merged |= c
                        else:
                            kept.append(c)
                    kept.append(merged)
                    components_of[with_node] = kept

        # Only the *unique connected components* are converted to node-tuples
        # (these are what the worth / NN forward consumes, and there are far
        # fewer distinct components than coalitions). The per-coalition tuples —
        # one length-≤N tuple for every one of the ~2·samples·N queried
        # coalitions, plus the coalition→worth dict — are intentionally NOT
        # materialised: that was O(samples·N²) host RAM and blew up on large
        # molecules (hundreds of atoms). The value loop is fully bitmask-native
        # (see :meth:`_finish_sampling_worth_by_mask`), so those tuples are never
        # needed.
        unique_component_masks = set()
        has_empty = False
        for mask, comps in components_of.items():
            if mask == 0:
                has_empty = True
            unique_component_masks.update(comps)
        component_tuple_of = {c: self._mask_to_labels(c, index_to_label)
                              for c in unique_component_masks}

        self._node_bits = node_bits
        self._random_node_bit = random_node_bit
        self._base_masks = base_masks
        # Integer-keyed structures only (no per-coalition tuples); both are freed
        # in :meth:`_finish_sampling_worth_by_mask` once `_worth_by_mask` exists.
        self._components_of = components_of
        self._component_tuple_of = component_tuple_of

        # The distinct connected subgraphs whose worth must be evaluated (bounded
        # by the number of components, not coalitions). The empty component () is
        # appended when some sampled coalition is empty, so the caller supplies
        # its (zero) worth exactly as before.
        components = list(component_tuple_of.values())
        if has_empty:
            components.append(())
        self.graph_restricted_coalitions = set(components)
        return components

    def _finish_sampling_worth_by_mask(self,
            graph_restricted_coalitions_to_worth: dict) -> dict:
        """Given worths for the unique connected components (computed
        single-molecule), assemble the bitmask-keyed worth table the value loop
        consumes.

        Fully bitmask-native: ``_worth_by_mask[m]`` is the sum of the worths of
        ``m``'s connected components, summed straight from the integer
        ``components_of`` map — no per-coalition node-tuples or coalition→worth
        dict are built (that materialisation was O(samples·N²) host RAM). Works
        for scalar worths and for tensor (multi-output) worths alike (the
        per-component worths are simply added). The component map is released
        afterwards, leaving only ``_worth_by_mask`` for the value loop.
        """
        self.graph_restricted_coalitions_to_worth = \
            graph_restricted_coalitions_to_worth
        worth_of_component_mask = {
            c: graph_restricted_coalitions_to_worth[t]
            for c, t in self._component_tuple_of.items()}
        # Worth of the empty coalition: the () component is included in the
        # evaluated set whenever some sampled coalition is empty, matching the
        # previous map_coalition_to_worth behaviour (pure game: coalition_function
        # of (); GNN: the explainer's zero `_empty_worth`).
        empty_worth = graph_restricted_coalitions_to_worth.get(())

        worth_by_mask = {}
        for mask, comps in self._components_of.items():
            if mask == 0:
                worth_by_mask[mask] = empty_worth
                continue
            it = iter(comps)
            total = worth_of_component_mask[next(it)]
            for c in it:
                total = total + worth_of_component_mask[c]
            worth_by_mask[mask] = total
        self._worth_by_mask = worth_by_mask

        # Release the (potentially large) integer component map; the value loop
        # only needs `_worth_by_mask`.
        self._components_of = None
        self._component_tuple_of = None
        return self._worth_by_mask

    def sample_all_mappings(self) -> None:
        """Samples the base permutations and the worths of every coalition the
        estimator will query, as class attributes:

            * `self.random_node` (int)
            * `self.permutations_without_random_node` (list[np.ndarray])
            * `self.graph_restricted_coalitions` (set[tuple]) — the *distinct
              connected components* whose worth was evaluated.
            * `self.graph_restricted_coalitions_to_worth` (dict)
            * bitmask helpers (`self._base_masks`, `self._node_bits`,
              `self._random_node_bit`, `self._worth_by_mask`)

        Note: the full per-coalition mappings (`self.coalitions`,
        `self.coalitions_to_graph_restricted_coalitions`,
        `self.coalitions_to_worth`) are deliberately *not* materialised — they
        cost O(samples·N²) host RAM and are unnecessary for the bitmask-native
        value loop. Use the exact :class:`MyersonCalculator` if you need them.
        """
        self.random_node, self.permutations_without_random_node \
            = self._sample_base_permutations(self.number_of_samples)
        self._build_sampling_worth_by_mask()

    def sample_all_myerson_values(self) -> np.ndarray:
        """Use Monte Carlo sampling to approximate the Myerson values for every
        node / player in the graph.

        Returns:
            np.ndarray: Sampled Myerson values for each node.
        """
        self.sample_all_mappings()
        return self._sampled_myerson_values_from_prepared()

    def _sampled_myerson_values_from_prepared(self) -> np.ndarray:
        """Run the integer-bitmask marginal-contribution value loop using the
        already-prepared worth table (``self._worth_by_mask`` etc.).
        """
        self.log.info("Calculating sampled Myerson values.")
        worth = self._worth_by_mask
        node_bits = self._node_bits
        random_node_bit = self._random_node_bit
        n = len(node_bits)

        # Integer-bitmask value loop: each iteration is a few int ops + two dict
        # lookups (no per-iteration array copy / sort / tuple construction).
        acc = [0.0] * n
        for mask in tqdm(self._base_masks, disable=self.disable_tqdm,
                         desc="Calculate sampled Myerson values"):
            for j in range(n):
                bit = node_bits[j]
                without = ((mask ^ bit) | random_node_bit) if (mask & bit) else mask
                acc[j] += worth[without | bit] - worth[without]

        my_values = np.array(acc, dtype=float) / self.number_of_samples
        log_string = "".join([f"\t{node}: {val:.4f}\n" for node, val in zip(self.grand_coalition, my_values, strict=True)])
        self.log.info(f"Sampled Myerson Values:\n{log_string}")
        return my_values

    def calculate_all_myerson_values(self) -> None:
        """Not implemented for sampling class.

        Raises:
            NotImplementedError: The MyersonSampler only has the
                `sample_all_myerson_values()` method. To accurately calculate the
                Myerson values, please use the `MyersonCalculator` class.
        """
        raise NotImplementedError("""The MyersonSampler only has the `sample_all_myerson_values()` method.
                     To accurately calculate the Myerson values,
                     please use the `MyersonCalculator` class.""")