"""Correctness of the computation paths: brute-force cross-checks, the
multi-output and legacy lattice paths, edge cases, and sampler behaviour
(agreement with the exact values and statistical guarantees)."""
import pytest
import numpy as np
import networkx as nx
from myerson import MyersonBudgetExceeded, MyersonCalculator, MyersonSampler
from .myerson_helpers import (brute_force_myerson, make_weighted_worth, make_multioutput_worth, _legacy_lattice_myerson, _reference_graphs)


class TestExactCorrectnessAgainstBruteForce:
    """Cross-check the production connected-subgraph enumeration against an
    independent ``2^N`` brute-force reference on a variety of topologies.
    """

    @pytest.mark.parametrize("name,graph", list(_reference_graphs().items()))
    def test_matches_brute_force(self, name, graph):
        """Exact values match the brute-force oracle across all topologies."""
        worth = make_weighted_worth()
        calc = MyersonCalculator(graph=graph, coalition_function=worth)
        got = calc.calculate_all_myerson_values()
        expected = brute_force_myerson(graph, worth)
        assert np.allclose(got, expected, atol=1e-9), (
            f"{name}: {got=} {expected=}")


class TestExactMultiOutput:
    """The tensor / multi-output accumulation in the connected-enum path
    (``_coerce_worth`` non-scalar) checked without any GNN dependency.
    """

    @pytest.mark.parametrize("name,graph", [
        ("path5", nx.path_graph(5)),
        ("cycle6", nx.cycle_graph(6)),
        ("gnp_7", nx.gnp_random_graph(7, 0.4, seed=2)),
    ])
    def test_matches_brute_force(self, name, graph):
        """Vector-valued (multi-output) worths match the brute-force oracle,
        with the expected ``(n_nodes, n_tasks)`` output shape.
        """
        worth = make_multioutput_worth(n_tasks=3)
        calc = MyersonCalculator(graph=graph, coalition_function=worth)
        got = calc.calculate_all_myerson_values()
        expected = brute_force_myerson(graph, worth)
        assert got.shape == (graph.number_of_nodes(), 3), f"{name}: {got.shape=}"
        assert np.allclose(got, expected, atol=1e-9), (
            f"{name}: {got=} {expected=}")


class TestLegacyLatticePathAgrees:
    """The legacy full-lattice path (`calculate_all_mappings` +
    `calculate_single_myerson_value`) must agree with the production
    connected-subgraph enumeration.
    """

    @pytest.mark.parametrize("name,graph", [
        ("path5", nx.path_graph(5)),
        ("cycle5", nx.cycle_graph(5)),
        ("star4", nx.star_graph(4)),
        ("complete4", nx.complete_graph(4)),
        ("gnp_6", nx.gnp_random_graph(6, 0.5, seed=1)),
    ])
    def test_lattice_matches_connected_enum(self, name, graph):
        """The legacy lattice path and the connected-enum path give equal
        values on the same games.
        """
        worth = make_weighted_worth()
        production = MyersonCalculator(
            graph=graph, coalition_function=worth).calculate_all_myerson_values()
        legacy = _legacy_lattice_myerson(
            MyersonCalculator(graph=graph, coalition_function=worth))
        assert np.allclose(production, legacy, atol=1e-9), (
            f"{name}: {production=} {legacy=}")


class TestExactEdgeCases:
    """Degenerate graphs: empty, single node, and all-isolated nodes."""

    def test_empty_graph_returns_empty_array(self):
        """A graph with no nodes yields an empty value array."""
        calc = MyersonCalculator(
            graph=nx.Graph(), coalition_function=make_weighted_worth())
        values = calc.calculate_all_myerson_values()
        assert values.shape == (0,)

    def test_single_node_graph(self):
        """The only coalition is the singleton, so its value is its worth."""
        graph = nx.Graph()
        graph.add_node(0)
        worth = make_weighted_worth()
        calc = MyersonCalculator(graph=graph, coalition_function=worth)
        values = calc.calculate_all_myerson_values()
        assert values.shape == (1,)
        assert values[0] == pytest.approx(worth((0,), graph), abs=1e-9)

    def test_all_isolated_nodes_equal_singleton_worths(self):
        """No edges: a node can never gain from a coalition, so MV_i = v({i})."""
        graph = nx.Graph()
        graph.add_nodes_from(range(4))
        worth = make_weighted_worth()
        calc = MyersonCalculator(graph=graph, coalition_function=worth)
        values = calc.calculate_all_myerson_values()
        expected = np.array([worth((i,), graph) for i in range(4)])
        assert np.allclose(values, expected, atol=1e-9)
        assert np.allclose(values, brute_force_myerson(graph, worth), atol=1e-9)


class TestSamplerAgreesWithExact:
    """Cross-validate the Monte-Carlo sampler against the exact values on a
    graph not covered by the analytic gloves cases.
    """

    def test_sampler_converges_to_exact(self):
        """With many samples the estimate converges to the exact values."""
        graph = _reference_graphs()["ring_with_tail"]
        worth = make_weighted_worth(scale=0.1)  # keep values O(1) for a tight tol
        exact = MyersonCalculator(
            graph=graph, coalition_function=worth).calculate_all_myerson_values()
        sampler = MyersonSampler(
            graph=graph, coalition_function=worth,
            number_of_samples=20000, seed=42, disable_tqdm=True)
        sampled = sampler.sample_all_myerson_values()
        assert np.allclose(sampled, exact, atol=1e-1), (
            f"{sampled=} {exact=}")

    def test_sampler_is_deterministic_for_fixed_seed(self):
        """A fixed seed yields bitwise-identical results across runs."""
        graph = nx.path_graph(5)
        worth = make_weighted_worth(scale=0.1)
        kwargs = dict(graph=graph, coalition_function=worth,
                      number_of_samples=500, seed=7, disable_tqdm=True)
        first = MyersonSampler(**kwargs).sample_all_myerson_values()
        second = MyersonSampler(**kwargs).sample_all_myerson_values()
        assert np.array_equal(first, second)


class TestExactInLoopBudget:
    """The budget can also trip *during* enumeration (not only in the preflight
    lower-bound check). A dense graph whose spanning-forest lower bound is below
    the budget but whose true connected-subgraph count exceeds it hits that
    branch.
    """

    def test_in_loop_budget_raises(self):
        """A budget above the preflight lower bound but below the true count
        must still abort, this time from inside the enumeration loop.
        """
        graph = nx.complete_graph(6)  # 2^6 - 1 = 63 connected subgraphs
        budget = 40
        calc = MyersonCalculator(
            graph=graph, coalition_function=make_weighted_worth())
        # Lower bound (37 for K6) is below the budget, so the preflight passes;
        # the abort therefore comes from generating the 41st connected subgraph.
        assert calc.connected_subgraph_count_lower_bound() <= budget
        calc._connected_enum_max_subgraphs = budget
        with pytest.raises(MyersonBudgetExceeded):
            calc.calculate_all_myerson_values()


class TestSamplerStatistics:
    """Statistical guarantees of the Monte-Carlo sampler beyond determinism."""

    def test_unbiased_average_over_seeds_approaches_exact(self):
        """Averaging estimates over many seeds approaches the exact values
        (the sampler is unbiased).
        """
        graph = nx.path_graph(5)
        worth = make_weighted_worth(scale=0.1)
        exact = MyersonCalculator(
            graph=graph, coalition_function=worth).calculate_all_myerson_values()

        estimates = []
        for seed in range(24):
            sampler = MyersonSampler(
                graph=graph, coalition_function=worth,
                number_of_samples=400, seed=seed, disable_tqdm=True)
            estimates.append(sampler.sample_all_myerson_values())
        mean_estimate = np.mean(estimates, axis=0)
        assert np.allclose(mean_estimate, exact, atol=2e-2), (
            f"{mean_estimate=} {exact=}")

    def test_error_shrinks_with_more_samples(self):
        """The mean absolute error decreases as the sample count grows."""
        graph = nx.path_graph(5)
        worth = make_weighted_worth(scale=0.1)
        exact = MyersonCalculator(
            graph=graph, coalition_function=worth).calculate_all_myerson_values()

        def mean_abs_error(n_samples):
            errors = []
            for seed in range(12):
                est = MyersonSampler(
                    graph=graph, coalition_function=worth,
                    number_of_samples=n_samples, seed=seed,
                    disable_tqdm=True).sample_all_myerson_values()
                errors.append(np.abs(est - exact).mean())
            return float(np.mean(errors))

        assert mean_abs_error(2000) < mean_abs_error(50)
