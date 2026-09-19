import pytest
import numpy as np
import torch
from myerson.pyg_explain import PerturbationExplainer, PerturbationClassExplainer
from .testmodels import GATConvModel
from .utils import rename_state_dict_keys, create_complete_graph_edges


@pytest.fixture(scope="module")
def device():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


@pytest.fixture(scope="module")
def regression_setup(device):
    torch.manual_seed(0)
    model = GATConvModel(dim_in=5, dim_out=1)
    with open("tests/testgraph.pt", "rb") as f:
        graph = torch.load(f, map_location=torch.device(device), weights_only=False)
    graph.edge_index = create_complete_graph_edges(graph.x.shape[0])
    solution = np.array([
                -0.11366,
                -0.11366,
                -0.11123,
                -0.14156,
                -0.11366,
                -0.11366,
                -0.11123,
                -0.07453,
                -0.12062,
                -0.07787,
                ])
    return model, graph, solution


@pytest.fixture(scope="module")
def classification_setup(device):
    torch.manual_seed(0)
    model = GATConvModel(dim_in=5, dim_out=3)
    with open("tests/testgraph.pt", "rb") as f:
        graph = torch.load(f, map_location=torch.device(device), weights_only=False)
    graph.edge_index = create_complete_graph_edges(graph.x.shape[0])
    solution = np.array([
                [-0.1137,  0.0520, -0.0139],
                [-0.1137,  0.0520, -0.0139],
                [-0.1112,  0.0462, -0.0268],
                [-0.1416,  0.0737, -0.0028],
                [-0.1137,  0.0520, -0.0139],
                [-0.1137,  0.0520, -0.0139],
                [-0.1112,  0.0462, -0.0268],
                [-0.0745,  0.0644, -0.0214],
                [-0.1206,  0.0754, -0.0202],
                [-0.0779,  0.0699, -0.0080],
                ])
    return model, graph, solution


class TestPerturbationExplainer:

    def test_complete_explanations(self, regression_setup):
        model, graph, solution = regression_setup
        explainer = PerturbationExplainer(graph, model, disable_tqdm=True)
        perturbation_values = explainer.calculate_all_perturbation_values()
        for xp, sol in zip(perturbation_values, solution):
            assert xp == pytest.approx(sol, abs=1e-5), f"{perturbation_values=}, {solution=}"


class TestPerturbationClassExplainer:

    def test_complete_explanations(self, classification_setup):
        import logging
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s - %(levelname)s] %(message)s', force=True)
        model, graph, solution = classification_setup
        explainer = PerturbationClassExplainer(graph, model, disable_tqdm=True)
        perturbation_values = explainer.calculate_all_perturbation_values()
        for xp, sol in zip(perturbation_values, solution):
            assert np.allclose(xp, sol, atol=0.01), f"{perturbation_values=}, {solution=}"