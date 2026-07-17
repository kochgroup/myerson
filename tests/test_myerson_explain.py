import pytest
import numpy as np
import torch
from myerson.pyg_explain import MyersonExplainer, MyersonClassExplainer
from myerson.pyg_explain import MyersonSamplingExplainer, MyersonSamplingClassExplainer
from .testmodels import GATConvModel
from .utils import rename_state_dict_keys


@pytest.fixture(scope="module")
def device():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


@pytest.fixture(scope="module")
def regression_setup(device):
    model = GATConvModel(dim_in=5, dim_out=1)
    with open("tests/testmodelparams.ckpt", "rb") as f:
        params = torch.load(f, map_location=torch.device(device), weights_only=True)
    state_dict = rename_state_dict_keys(params['state_dict'], device)
    model.load_state_dict(state_dict)
    with open("tests/testgraph.pt", "rb") as f:
        graph = torch.load(f, map_location=torch.device(device), weights_only=False)

    solution = np.array([
        0.3334993031053314,
        0.32414250585531335,
        0.09926704892090386,
        -0.235700370016553,
        0.32414250134948625,
        0.3334992930174817,
        0.2064565007175723,
        0.14213201623587385,
        -0.12044816162614626,
        0.11972642920556467
        ])
    
    return model, graph, solution


@pytest.fixture(scope="module")
def classification_setup(device):
    torch.manual_seed(0)
    model = GATConvModel(dim_in=5, dim_out=3)
    last_params = torch.tensor([0.0366, -0.0206, -0.0398])
    assert torch.allclose(last_params, list(model.parameters())[-1], atol=0.0001), f"{last_params=}, {list(model.parameters())[-1].detach()=}"
    with open("tests/testgraph.pt", "rb") as f:
        graph = torch.load(f, map_location=torch.device(device), weights_only=False)

    solution = np.array([
        [-0.1079,  0.0471, -0.0231],
        [-0.1079,  0.0464, -0.0229],
        [-0.1230,  0.0496, -0.0087],
        [-0.1097,  0.0562, -0.0421],
        [-0.1079,  0.0464, -0.0229],
        [-0.1079,  0.0471, -0.0231],
        [-0.1090,  0.0511, -0.0247],
        [-0.0823,  0.0611, -0.0204],
        [-0.1285,  0.0739, -0.0161],
        [-0.0874,  0.0686, -0.0289]
        ])

    return model, graph, solution


class TestMyersonExplainer:

    # def test_with_fast_restrict(self):
    #     model, graph, solution = regression_setup
    #     explainer = MyersonExplainer(graph, model)
    #     if platform.system == 'Linux':
    #         assert explainer.fast_restrict_available == True
    #     my_values = explainer.calculate_all_myerson_values()
    #     for my, sol in zip(my_values.values(), solution.values()):
    #         assert my == pytest.approx(sol, abs=1e-5), f"{my_values=}, {solution=}"

    def test_with_restrict(self, regression_setup):
        model, graph, solution = regression_setup
        explainer = MyersonExplainer(graph, model)
        # explainer.set_restrict(use_fast_restrict=False)
        my_values = explainer.calculate_all_myerson_values()
        for my, sol in zip(my_values, solution):
            assert my == pytest.approx(sol, abs=1e-5), f"{my_values=}, {solution=}"


class TestMyersonSamplingExplainer:

    # def test_with_fast_restrict(self):
    #     model, graph, solution = regression_setup
    #     sampler = MyersonSamplingExplainer(graph, model, seed=42)
    #     if platform.system == 'Linux':
    #         assert sampler.fast_restrict_available == True
    #     my_values = sampler.sample_all_myerson_values()
    #     for my, sol in zip(my_values.values(), solution.values()):
    #         assert my == pytest.approx(sol, abs=1e-1), f"{my_values=}, {solution=}"

    def test_with_restrict(self, regression_setup):
        model, graph, solution = regression_setup
        sampler = MyersonSamplingExplainer(graph, model, seed=42)
        # sampler.set_restrict(use_fast_restrict=False)
        my_values = sampler.sample_all_myerson_values()
        for my, sol in zip(my_values, solution):
            assert my == pytest.approx(sol, abs=1e-1), f"{my_values=}, {solution=}"


class TestMyersonClassExplainer:

    def test_with_restrict(self, classification_setup):
        model, graph, solution = classification_setup
        explainer = MyersonClassExplainer(graph, model)
        # sampler.set_restrict(use_fast_restrict=False)
        my_values = explainer.calculate_all_myerson_values()
        for my, sol in zip(my_values, solution):
            assert np.allclose(my, sol, atol=0.0001), f"{my_values=}, {solution=}"


class TestMyersonSamplingClassExplainer:

    def test_with_restrict(self, classification_setup):
        model, graph, solution = classification_setup
        sampler = MyersonSamplingClassExplainer(graph, model, seed=42)
        # sampler.set_restrict(use_fast_restrict=False)
        my_values = sampler.sample_all_myerson_values()
        for my, sol in zip(my_values, solution):
            assert np.allclose(my, sol, atol=0.01), f"{my_values=}, {solution=}"


class TestBatchingEquivalence:
    """The batched worth path (``_batch_data_from_coalitions`` +
    ``calculate_worth_of_graph_restricted_coalitions``) and the single-subgraph
    path (``subgraph_from_coalition`` +
    ``calculate_worth_of_single_graph_restricted_coalition``) reimplement the same
    masking/relabelling. These tests guard against the two code paths drifting.
    """

    @staticmethod
    def _coalitions(n_nodes):
        return [
            (0,),
            (n_nodes - 1,),
            (0, 1),
            (0, 2, 4),
            (1, 3, 5, 7),
            tuple(range(n_nodes)),
        ]

    def test_regression_paths_agree(self, regression_setup):
        model, graph, _ = regression_setup
        explainer = MyersonExplainer(graph, model)
        n_nodes = graph.x.shape[0]
        coalitions = self._coalitions(n_nodes)

        batched = explainer.calculate_worth_of_graph_restricted_coalitions(coalitions)
        for c in coalitions:
            single = explainer.calculate_worth_of_single_graph_restricted_coalition(
                c, graph)
            assert batched[c] == pytest.approx(single, abs=1e-6), (
                f"batched vs single mismatch for {c}: {batched[c]} != {single}")

    def test_classification_paths_agree(self, classification_setup):
        model, graph, _ = classification_setup
        explainer = MyersonClassExplainer(graph, model)
        n_nodes = graph.x.shape[0]
        coalitions = self._coalitions(n_nodes)

        batched = explainer.calculate_worth_of_graph_restricted_coalitions(coalitions)
        for c in coalitions:
            single = explainer.calculate_worth_of_single_graph_restricted_coalition(
                c, graph)
            assert np.allclose(batched[c], single, atol=1e-6), (
                f"batched vs single mismatch for {c}: {batched[c]} != {single}")

    def test_batch_data_matches_single_subgraph(self, regression_setup):
        """Tensor-level check: the disjoint-union built by
        ``_batch_data_from_coalitions`` yields, per graph, the same forward output
        as individually collated ``subgraph_from_coalition`` graphs.
        """
        model, graph, _ = regression_setup
        explainer = MyersonExplainer(graph, model)
        n_nodes = graph.x.shape[0]
        coalitions = self._coalitions(n_nodes)

        x, edge_index, batch = explainer._batch_data_from_coalitions(coalitions)
        batched_out = explainer._forward(x, edge_index, batch)

        for i, c in enumerate(coalitions):
            subgraph = explainer.subgraph_from_coalition(c, graph)
            single_out = explainer._forward(
                subgraph.x, subgraph.edge_index, explainer._batch_var(subgraph))
            assert torch.allclose(batched_out[i], single_out.squeeze(0), atol=1e-6), (
                f"forward mismatch for coalition {c}")