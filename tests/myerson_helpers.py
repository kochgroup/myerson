"""Shared, dependency-light test helpers for the Myerson core.

Kept free of torch/PyG imports so it can be used by the pure game-theory test
suite. Provides an independent brute-force oracle, coalition-function factories
(additive, mass-based, topological descriptors, unanimity, ...), a small library
of reference graphs, and misc utilities used across the ``test_myerson_*`` files.
"""
from itertools import combinations
from math import factorial

import numpy as np
import networkx as nx

from myerson import MyersonCalculator


def brute_force_myerson(graph, coalition_function):
    r"""Reference Myerson values from the definition, independent of the package.

    The Myerson value is the Shapley value of the *graph-restricted* game
    :math:`v'(S) = \sum_{C \in \text{components}(S)} v(C)`, where the components
    are the connected components of the subgraph induced by ``S``. Connectivity
    is computed with NetworkX (not the package's bitmask machinery) so this is a
    genuinely independent oracle:

    .. math::

        \text{MV}_i = \sum_{S \subseteq N \setminus \{i\}}
            \frac{|S|!\,(|N|-|S|-1)!}{|N|!}\,\big(v'(S \cup \{i\}) - v'(S)\big)

    Iterates the full ``2^N`` subset lattice, so only use it on small graphs.
    Returns values in ``graph.nodes()`` order and supports scalar or vector
    (multi-output) worths alike.
    """
    nodes = list(graph.nodes())
    n = len(nodes)

    def v_restricted(subset):
        # v'(∅) = 0 (a sum over zero components).
        if not subset:
            return 0.0
        total = None
        for comp in nx.connected_components(graph.subgraph(subset)):
            worth = coalition_function(tuple(sorted(comp)), graph)
            total = worth if total is None else total + worth
        return total

    values = []
    for i in nodes:
        others = [x for x in nodes if x != i]
        mv = 0.0
        for r in range(len(others) + 1):
            for subset in combinations(others, r):
                s = len(subset)
                weight = factorial(s) * factorial(n - s - 1) / factorial(n)
                with_i = v_restricted(set(subset) | {i})
                without_i = v_restricted(set(subset))
                mv = mv + weight * (with_i - without_i)
        values.append(mv)
    return np.array(values)

def make_weighted_worth(scale: float = 1.0):
    """A deterministic, non-additive scalar worth: ``(sum of node weights)^2``.

    Node ``k`` has weight ``scale * (k + 1)``. Squaring makes the worth
    genuinely super-additive within a component, so the Skibski neighbour-set
    weighting is actually exercised (a purely additive worth would hide sign or
    weighting mistakes).
    """
    def worth(coalition, nx_graph):
        if not coalition:
            return 0.0
        return float(scale * sum(node + 1 for node in coalition)) ** 2
    return worth

def make_multioutput_worth(n_tasks: int = 3):
    """Vector-valued worth (one non-additive channel per task)."""
    def worth(coalition, nx_graph):
        if not coalition:
            return np.zeros(n_tasks, dtype=float)
        base = sum(node + 1 for node in coalition)
        return np.array([float(base) ** (k + 1) for k in range(n_tasks)])
    return worth

_ATOM_WEIGHTS = [12.011, 15.999, 14.007, 1.008, 32.06, 30.974, 18.998, 35.45]

def _weights_for(graph) -> dict:
    """Assign a distinct positive weight to each node, in node order."""
    return {node: _ATOM_WEIGHTS[i % len(_ATOM_WEIGHTS)]
            for i, node in enumerate(graph.nodes())}

def make_additive_worth(weights: dict):
    """Additive worth ``v(S) = sum of node weights`` (e.g. molecular weight)."""
    def worth(coalition, nx_graph):
        return float(sum(weights[node] for node in coalition))
    return worth

def make_mass_fraction_worth(weights: dict, total: float):
    """Additive worth normalised by a constant molecule total: ``v(S)/total``."""
    def worth(coalition, nx_graph):
        return float(sum(weights[node] for node in coalition)) / total
    return worth

def make_component_normalized_worth():
    """Each connected component contributes its own mass divided by itself = 1.

    So ``v(C) = 1`` for any non-empty component and the graph-restricted game
    ``v'(S)`` equals the number of connected components of ``S`` — a genuinely
    non-additive, topology-dependent game.
    """
    def worth(coalition, nx_graph):
        return 1.0 if coalition else 0.0
    return worth

def make_atsc0_worth(weights: dict):
    """Centered Moreau-Broto autocorrelation, lag 0 (the ATSC0 descriptor).

    ``v(S) = sum_i (m_i - mean(m))^2 = sum m_i^2 - (sum m_i)^2 / |S|``, evaluated
    per connected component. The ``/|S|`` term makes this non-additive *and*
    size-dependent: a node's marginal contribution can be negative and change
    sign as the coalition grows, so it exercises the neighbour-set weighting
    (including its negative terms) more thoroughly than a polynomial of the sum.
    Singletons and the empty set are worth 0.
    """
    def worth(coalition, nx_graph):
        if len(coalition) <= 1:
            return 0.0
        masses = np.array([weights[node] for node in coalition], dtype=float)
        centered = masses - masses.mean()
        return float((centered ** 2).sum())
    return worth

def make_sum_squared_masses_worth(weights: dict):
    """Additive component ``A(S) = sum_i m_i^2`` of the ATSC0 decomposition."""
    def worth(coalition, nx_graph):
        return float(sum(weights[node] ** 2 for node in coalition))
    return worth

def make_sumsq_over_size_worth(weights: dict):
    """Non-additive component ``B(S) = (sum_i m_i)^2 / |S|`` of ATSC0."""
    def worth(coalition, nx_graph):
        if not coalition:
            return 0.0
        return float(sum(weights[node] for node in coalition)) ** 2 / len(coalition)
    return worth

def _legacy_lattice_myerson(calc: MyersonCalculator) -> np.ndarray:
    """Drive the legacy full-lattice path (``calculate_all_mappings`` +
    ``calculate_single_myerson_value``) to produce Myerson values, mirroring how
    ``ShapleyCalculator.calculate_all_shapley_values`` drives its own lattice.
    """
    calc.calculate_all_mappings()
    return np.array([
        calc.calculate_single_myerson_value(
            node, calc.grand_coalition, calc.coalitions, calc.coalitions_to_worth)
        for node in calc.grand_coalition
    ])

def gloves_game_coalition_function(coalition: tuple,
                                   nx_graph: nx.classes.graph.Graph) -> float:
    """Coalition function for the gloves game.

    Args:
        coalition (tuple): The coalition for which to calculate the payoff
            of the game.
        nx_graph (nx.classes.graph.Graph): For this implementation of the
            gloves game, we expect a networkX graph which has nodes with a
            `glove` attribute that can be either `right` or `left`.

    Returns:
        float: Worth of the coalition.
    """
    if len(coalition) <= 1:
        return 0.

    gloves = nx.get_node_attributes(nx_graph, 'glove')
    r = sum([1 for k, v in gloves.items() if (v=="right" and k in coalition)])
    l = sum([1 for k, v in gloves.items() if (v=="left" and k in coalition)])
    return float(min(r, l))

def _reference_graphs():
    graphs = {
        "path5": nx.path_graph(5),
        "cycle6_benzene": nx.cycle_graph(6),
        "star4": nx.star_graph(4),
        "complete4": nx.complete_graph(4),
        "two_triangles_disconnected": nx.disjoint_union(
            nx.complete_graph(3), nx.complete_graph(3)),
        "balanced_tree": nx.balanced_tree(2, 2),
    }
    # Ring (5-cycle) with a two-node tail hanging off node 0.
    ring_with_tail = nx.cycle_graph(5)
    ring_with_tail.add_edges_from([(0, 5), (5, 6)])
    graphs["ring_with_tail"] = ring_with_tail
    # A few reproducible random graphs of increasing size.
    for i, (n, p, seed) in enumerate([(6, 0.5, 1), (7, 0.4, 2), (8, 0.35, 3)]):
        graphs[f"gnp_{n}_{seed}"] = nx.gnp_random_graph(n, p, seed=seed)
    return graphs

def zagreb_m1_worth(coalition, nx_graph):
    """First Zagreb index ``M1 = sum_i deg(i)^2`` over the induced subgraph."""
    sub = nx_graph.subgraph(coalition)
    return float(sum(deg * deg for _, deg in sub.degree()))

def wiener_index_worth(coalition, nx_graph):
    """Wiener index = sum of shortest-path distances over all node pairs."""
    sub = nx_graph.subgraph(coalition)
    if sub.number_of_nodes() < 2:
        return 0.0
    total = 0
    for _, dists in nx.all_pairs_shortest_path_length(sub):
        total += sum(dists.values())
    return float(total) / 2.0

def eccentric_connectivity_worth(coalition, nx_graph):
    """Eccentric connectivity index ``= sum_i deg(i) * ecc(i)``.

    ``ecc(i)`` is the largest topological distance from node ``i`` to any other
    node in the subgraph.
    """
    sub = nx_graph.subgraph(coalition)
    if sub.number_of_nodes() < 2:
        return 0.0
    ecc = nx.eccentricity(sub)
    return float(sum(sub.degree(node) * ecc[node] for node in sub.nodes()))

def randic_chi_worth(coalition, nx_graph):
    """Randic connectivity index ``= sum_{(u,v) in edges} 1/sqrt(deg(u)*deg(v))``.

    (The classic first-order molecular connectivity index ``1-chi``, a product
    of node degrees over bonds.)
    """
    sub = nx_graph.subgraph(coalition)
    total = 0.0
    for u, v in sub.edges():
        total += 1.0 / (sub.degree(u) * sub.degree(v)) ** 0.5
    return total

def edge_count_worth(coalition, nx_graph):
    """Number of edges in the induced subgraph, ``v(S) = |E(S)|``.

    Decomposes into one 2-player unanimity game per edge, so fairness splits
    each edge 50/50 and the Myerson value has the exact closed form
    ``MV_i = deg(i) / 2``.
    """
    return float(nx_graph.subgraph(coalition).number_of_edges())

_TOPOLOGICAL_DESCRIPTORS = {
    "zagreb_m1": zagreb_m1_worth,
    "wiener": wiener_index_worth,
    "eccentric_connectivity": eccentric_connectivity_worth,
    "randic_chi": randic_chi_worth,
}

def make_weighted_sq_worth(weights: dict):
    """Non-additive worth ``v(S) = (sum of node weights)^2`` (per component)."""
    def worth(coalition, nx_graph):
        return float(sum(weights[node] for node in coalition)) ** 2
    return worth

def make_unanimity_worth(target):
    """Unanimity game ``u_T(S) = 1 if T subset of S else 0``.

    On a complete graph its Shapley (= Myerson) value is ``1/|T|`` for members
    of ``T`` and ``0`` otherwise -- an exact rational closed form.
    """
    target = frozenset(target)

    def worth(coalition, nx_graph):
        return 1.0 if target.issubset(coalition) else 0.0
    return worth

def make_linear_combo_worth(a, v, b, w):
    """``a*v(S) + b*w(S)`` -- used to probe linearity of the value operator."""
    def worth(coalition, nx_graph):
        return a * v(coalition, nx_graph) + b * w(coalition, nx_graph)
    return worth

def _values_by_node(calc, graph):
    """Map ``node -> Myerson value`` (calculator returns node-order arrays)."""
    values = calc.calculate_all_myerson_values()
    return {node: values[i] for i, node in enumerate(graph.nodes())}
