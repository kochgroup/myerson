import pytest
import numpy as np
import torch
from myerson.pyg_explain import ShapleyExplainer, ShapleySamplingExplainer
from myerson.pyg_explain import ShapleyClassExplainer, ShapleySamplingClassExplainer
from myerson.pyg_explain import MyersonExplainer, MyersonClassExplainer
from .testmodels import GATConvModel
from .utils import rename_state_dict_keys, create_complete_graph_edges


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
    graph.edge_index = create_complete_graph_edges(graph.x.shape[0])
    myerson_explainer = MyersonExplainer(graph, model)
    solution = myerson_explainer.calculate_all_myerson_values()
    return model, graph, solution


@pytest.fixture(scope="module")
def classification_setup(device):
    import logging
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s - %(levelname)s] %(message)s', force=True)

    # setting multiclass output with random but reproducible weights
    torch.manual_seed(0)
    model = GATConvModel(dim_in=5, dim_out=3)
    last_params = torch.tensor([0.0366, -0.0206, -0.0398])
    assert(torch.allclose(last_params, list(model.parameters())[-1], atol=0.0001)), f"{last_params=}, {list(model.parameters())[-1].detach()=}"

    with open("tests/testgraph.pt", "rb") as f:
        graph = torch.load(f, map_location=torch.device(device), weights_only=False)
    graph.edge_index = create_complete_graph_edges(graph.x.shape[0])
    myerson_explainer = MyersonClassExplainer(graph, model)
    solution = myerson_explainer.calculate_all_myerson_values()
    return model, graph, solution


class TestShapleyExplainer:

    def test_complete_explanations(self, regression_setup):
        model, graph, solution = regression_setup
        explainer = ShapleyExplainer(graph, model, disable_tqdm=True)
        sh_values = explainer.calculate_all_shapley_values()
        for sh, sol in zip(sh_values, solution):
            assert sh == pytest.approx(sol, abs=1e-5), f"{sh_values=}, {solution=}"

    def test_sampled_explanations(self, regression_setup):
        model, graph, solution = regression_setup
        sampler = ShapleySamplingExplainer(graph, model, seed=42, number_of_samples=1000, disable_tqdm=True)
        sh_values = sampler.sample_all_shapley_values()
        for sh, sol in zip(sh_values, solution):
            assert sh == pytest.approx(sol, abs=1e-1), f"{sh_values=}, {solution=}"


class TestShapleyClassExplainer:
                       
    def test_complete_class_explanations(self, classification_setup):
        model, graph, solution = classification_setup
        explainer = ShapleyClassExplainer(graph, model, disable_tqdm=True)
        sh_values = explainer.calculate_all_shapley_values()
        for sh, sol in zip(sh_values, solution):
            assert np.allclose(sh, sol, atol=0.0001), f"{sh_values=}, {solution=}"

    def test_sampled_class_explanations(self, classification_setup):
        model, graph, solution = classification_setup
        explainer = ShapleySamplingClassExplainer(graph, model, disable_tqdm=True)
        sh_values = explainer.sample_all_shapley_values()
        for sh, sol in zip(sh_values, solution):
            assert np.allclose(sh, sol, atol=0.01), f"{sh_values=}, {solution=}"
