"""Game-theoretic axioms and properties of the Myerson value: efficiency and
component efficiency, fairness (balanced contributions), symmetry, the null
player and linearity axioms, locality, equality with the plain Shapley value on
complete graphs, and closed-form games."""
import pytest
import numpy as np
import networkx as nx
from myerson import MyersonCalculator, ShapleyCalculator
from .myerson_helpers import (brute_force_myerson, make_weighted_worth, _weights_for, make_atsc0_worth, _reference_graphs, zagreb_m1_worth, make_weighted_sq_worth, make_unanimity_worth, make_linear_combo_worth, _values_by_node)


class TestMyersonLocality:
    """The defining Myerson property: a node's value depends only on its own
    connected component (graph restriction), not on the rest of the graph.
    """

    def test_isolated_node_gets_its_standalone_worth(self):
        """An isolated node is always its own component, so its marginal is
        always ``v({i})`` and its Myerson value equals that standalone worth.
        """
        graph = nx.path_graph(3)
        graph.add_node(7)  # isolated
        worth = make_weighted_worth()  # v(S) = (sum node+1)^2
        calc = MyersonCalculator(graph=graph, coalition_function=worth)
        values = _values_by_node(calc, graph)
        assert values[7] == pytest.approx(worth((7,), graph), abs=1e-9)

    def test_adding_isolated_node_leaves_others_unchanged(self):
        """Adding a disconnected extra player must not change any other node's
        Myerson value.
        """
        base = nx.path_graph(4)
        worth = make_weighted_worth()
        base_vals = _values_by_node(
            MyersonCalculator(graph=base, coalition_function=worth), base)

        extended = base.copy()
        extended.add_node(99)  # disconnected extra player
        ext_vals = _values_by_node(
            MyersonCalculator(graph=extended, coalition_function=worth), extended)

        for node in base.nodes():
            assert ext_vals[node] == pytest.approx(base_vals[node], abs=1e-9), node

    def test_component_values_match_component_computed_alone(self):
        """A node's value must equal its value when its own component is
        computed in isolation (locality of the Myerson value).
        """
        comp_a = nx.path_graph(3)                       # nodes 0,1,2
        comp_b = nx.relabel_nodes(nx.cycle_graph(3),
                                  {0: 10, 1: 11, 2: 12})  # nodes 10,11,12
        combined = nx.union(comp_a, comp_b)
        worth = make_weighted_worth()

        combined_vals = _values_by_node(
            MyersonCalculator(graph=combined, coalition_function=worth), combined)
        a_vals = _values_by_node(
            MyersonCalculator(graph=comp_a, coalition_function=worth), comp_a)
        b_vals = _values_by_node(
            MyersonCalculator(graph=comp_b, coalition_function=worth), comp_b)

        for node in comp_a.nodes():
            assert combined_vals[node] == pytest.approx(a_vals[node], abs=1e-9)
        for node in comp_b.nodes():
            assert combined_vals[node] == pytest.approx(b_vals[node], abs=1e-9)


class TestMyersonEqualsShapleyOnCompleteGraph:
    """On a complete graph every coalition is connected, so the graph
    restriction is a no-op and the Myerson value must equal the plain Shapley
    value. Cross-validates ``myerson.py`` against the independent ``shapley.py``.
    """

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    @pytest.mark.parametrize("worth_factory", [
        lambda: make_weighted_worth(),
        lambda: make_atsc0_worth(_weights_for(nx.complete_graph(5))),
        lambda: zagreb_m1_worth,
    ])
    def test_myerson_matches_shapley(self, n, worth_factory):
        """On a complete graph the Myerson values equal the plain Shapley
        values (the graph restriction is a no-op).
        """
        graph = nx.complete_graph(n)
        worth = worth_factory()
        myerson = MyersonCalculator(
            graph=graph, coalition_function=worth).calculate_all_myerson_values()
        shapley = ShapleyCalculator(
            graph=graph, coalition_function=worth).calculate_all_shapley_values()
        assert np.allclose(myerson, shapley, atol=1e-9), f"{n=} {myerson=} {shapley=}"


class TestUnanimityGameClosedForm:
    """Exact rational closed form for the unanimity game."""

    @pytest.mark.parametrize("n,target", [
        (4, (0, 1)),
        (5, (0, 2, 4)),
        (5, (3,)),
        (6, (1, 2, 3, 4)),
    ])
    def test_complete_graph_matches_one_over_size(self, n, target):
        """On a complete graph the unanimity game gives ``1/|T|`` to each member
        of the target set and ``0`` to everyone else.
        """
        graph = nx.complete_graph(n)
        worth = make_unanimity_worth(target)
        values = MyersonCalculator(
            graph=graph, coalition_function=worth).calculate_all_myerson_values()
        expected = np.array(
            [1.0 / len(target) if node in target else 0.0
             for node in graph.nodes()])
        assert np.allclose(values, expected, atol=1e-9), f"{values=} {expected=}"

    def test_unanimity_on_path_matches_brute_force(self):
        """No closed form once connectivity restricts the coalitions, but the
        brute-force oracle still applies.
        """
        graph = nx.path_graph(5)
        worth = make_unanimity_worth((0, 4))
        got = MyersonCalculator(
            graph=graph, coalition_function=worth).calculate_all_myerson_values()
        expected = brute_force_myerson(graph, worth)
        assert np.allclose(got, expected, atol=1e-9), f"{got=} {expected=}"


class TestNullPlayerAxiom:
    """A player whose marginal contribution is always zero gets value zero."""

    def test_isolated_zero_worth_node_is_null(self):
        """Node 4 is isolated (never merges components) and has weight 0, so its
        singleton worth is 0 -> it contributes nothing to any coalition, even
        though the worth is non-additive for the connected players.
        """
        graph = nx.path_graph(4)
        graph.add_node(4)
        weights = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 0.0}
        worth = make_weighted_sq_worth(weights)
        values = _values_by_node(
            MyersonCalculator(graph=graph, coalition_function=worth), graph)
        assert values[4] == pytest.approx(0.0, abs=1e-9)


class TestValueOperatorLinearity:
    """The Myerson value is linear in the coalition function/game."""

    @pytest.mark.parametrize("graph_name,graph", [
        ("path5", nx.path_graph(5)),
        ("cycle6", nx.cycle_graph(6)),
        ("gnp_7", nx.gnp_random_graph(7, 0.4, seed=2)),
    ])
    def test_additivity_and_scaling(self, graph_name, graph):
        """The Myerson value is linear in the game: for any scalars ``a, b`` and
        coalition functions ``v, w``, ``MV(a*v + b*w) == a*MV(v) + b*MV(w)``.

        Using ``a > 0`` and ``b < 0`` on two structurally different worths (a
        weight-sum square and the first Zagreb index) exercises additivity and
        scaling together, including a negative coefficient.
        """
        v = make_weighted_worth()
        w = zagreb_m1_worth
        a, b = 2.5, -1.5

        mv_v = MyersonCalculator(
            graph=graph, coalition_function=v).calculate_all_myerson_values()
        mv_w = MyersonCalculator(
            graph=graph, coalition_function=w).calculate_all_myerson_values()
        mv_combo = MyersonCalculator(
            graph=graph,
            coalition_function=make_linear_combo_worth(a, v, b, w)
        ).calculate_all_myerson_values()

        assert np.allclose(mv_combo, a * mv_v + b * mv_w, atol=1e-9), (
            f"{graph_name}: {mv_combo=} {a*mv_v + b*mv_w=}")


class TestEfficiencyAxiom:
    """Efficiency: the Myerson values sum to the worth of the (graph-restricted)
    grand coalition.
    """

    @pytest.mark.parametrize("name,graph", [
        ("path5", nx.path_graph(5)),
        ("cycle6", nx.cycle_graph(6)),
        ("two_triangles_disconnected", nx.disjoint_union(
            nx.complete_graph(3), nx.complete_graph(3))),
        ("ring_with_tail", _reference_graphs()["ring_with_tail"]),
    ])
    def test_values_sum_to_grand_coalition_worth(self, name, graph):
        """Sum of the Myerson values equals the graph-restricted grand-coalition
        worth, on both connected and disconnected graphs.
        """
        worth = make_weighted_worth()
        calc = MyersonCalculator(graph=graph, coalition_function=worth)
        values = calc.calculate_all_myerson_values()
        grand = calc.calculate_worth_of_grand_coalition(graph)
        assert values.sum() == pytest.approx(grand, abs=1e-9), name


class TestComponentEfficiencyAxiom:
    """Component efficiency (a Myerson-characterizing axiom): within each
    connected component the values sum to that component's own worth -- strictly
    stronger than plain efficiency on disconnected graphs.
    """

    @pytest.mark.parametrize("name,graph", [
        ("two_triangles_disconnected", nx.disjoint_union(
            nx.complete_graph(3), nx.complete_graph(3))),
        ("path_plus_isolated", nx.disjoint_union(
            nx.path_graph(3), nx.empty_graph(1))),
        ("ring_with_tail", _reference_graphs()["ring_with_tail"]),
    ])
    def test_per_component_sum_equals_component_worth(self, name, graph):
        """For every connected component ``C``, ``sum_{i in C} MV_i == v(C)``."""
        worth = make_weighted_worth()
        calc = MyersonCalculator(graph=graph, coalition_function=worth)
        values = _values_by_node(calc, graph)
        for comp in nx.connected_components(graph):
            comp_nodes = tuple(sorted(comp))
            got = sum(values[node] for node in comp_nodes)
            assert got == pytest.approx(worth(comp_nodes, graph), abs=1e-9), (
                f"{name}: component {comp_nodes}")


class TestFairnessAxiom:
    """Fairness / balanced contributions (Myerson 1977): the two endpoints of an
    edge gain equally from that edge. For every edge ``(i, j)``,
    ``v_i(G) - v_i(G \\ {ij}) == v_j(G) - v_j(G \\ {ij})``. Together with
    component efficiency this uniquely characterizes the Myerson value.
    """

    @pytest.mark.parametrize("name,graph", [
        ("path5", nx.path_graph(5)),
        ("cycle6", nx.cycle_graph(6)),
        ("star4", nx.star_graph(4)),
        ("ring_with_tail", _reference_graphs()["ring_with_tail"]),
        ("gnp_7", nx.gnp_random_graph(7, 0.4, seed=2)),
    ])
    def test_balanced_contributions_across_edges(self, name, graph):
        """Removing an edge changes the values of its two endpoints by the same
        amount, for every edge in the graph.
        """
        worth = make_weighted_worth()
        full = _values_by_node(
            MyersonCalculator(graph=graph, coalition_function=worth), graph)
        for i, j in graph.edges():
            cut = graph.copy()
            cut.remove_edge(i, j)
            cut_vals = _values_by_node(
                MyersonCalculator(graph=cut, coalition_function=worth), cut)
            delta_i = full[i] - cut_vals[i]
            delta_j = full[j] - cut_vals[j]
            assert delta_i == pytest.approx(delta_j, abs=1e-9), (
                f"{name}: edge {(i, j)} {delta_i=} {delta_j=}")


class TestSymmetryAxiom:
    """Symmetry / equal treatment of equals: players that are interchangeable in
    the game receive equal values. A structural worth on a symmetric graph makes
    the corresponding nodes interchangeable.
    """

    def test_vertex_transitive_graph_gives_equal_values(self):
        """On a vertex-transitive graph every node is symmetric, so a structural
        worth assigns them all the same value.
        """
        graph = nx.cycle_graph(6)
        values = MyersonCalculator(
            graph=graph, coalition_function=zagreb_m1_worth
        ).calculate_all_myerson_values()
        assert np.allclose(values, values[0], atol=1e-9), f"{values=}"

    def test_mirror_symmetric_path_pairs_are_equal(self):
        """On a path the mirror map ``k -> n-1-k`` is an automorphism, so a
        structural worth gives mirror-paired nodes equal values.
        """
        graph = nx.path_graph(5)
        v = MyersonCalculator(
            graph=graph, coalition_function=zagreb_m1_worth
        ).calculate_all_myerson_values()
        assert v[0] == pytest.approx(v[4], abs=1e-9)
        assert v[1] == pytest.approx(v[3], abs=1e-9)
