import pytest

import numpy as np
import torch
from chemprop.models.model import MPNN
from chemprop.data import MoleculeDatapoint, MoleculeDataset

from myerson.chemprop_explain import (
    MyersonExplainer,
    MyersonSamplingExplainer,
    MyersonClassExplainer,
    MyersonSamplingClassExplainer
)



@pytest.fixture(scope="module")
def device():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


@pytest.fixture(scope="module")
def regression_setup(device):
    model = MPNN.load_from_checkpoint("tests/chemprop_regression.ckpt", map_location=device)
    dataset = MoleculeDataset([MoleculeDatapoint.from_smi("c1ccccc1O", [3.3])])
    graph = dataset[0].mg

    solution = np.array([
        0.20805272853800247,
        -0.2598040096816563,
        -0.618848564511254,
        -0.2598039262351536,
        0.20805272853800247,
        1.5619634935898452,
        2.326792989139044
        ])

    return model, graph, solution


@pytest.fixture(scope="module")
def classification_setup(device):
    model = MPNN.load_from_checkpoint("tests/chemprop_multitask.ckpt", map_location=device)
    dataset = MoleculeDataset([MoleculeDatapoint.from_smi("C#Cc1ccccc1O", [1, 0, 1])])
    graph = dataset[0].mg
    solution = np.array([
        [0.1456, 0.3633, 0.4215],
        [ 0.0160, -0.0136,  0.4900],
        [-0.0356, -0.2984,  0.0254],
        [-0.0075,  0.1749,  0.0371],
        [-0.0142,  0.1795,  0.0070],
        [-0.0165,  0.1813,  0.0044],
        [0.0187, 0.1490, 0.0131],
        [ 0.2241, -0.5621, -0.2310],
        [ 0.6695, -0.1739,  0.2326]
        ])

    return model, graph, solution


class TestMyersonExplainer:

    def test_with_restrict(self, regression_setup):
        model, graph, solution = regression_setup
        explainer = MyersonExplainer(graph, model)
        my_values = explainer.calculate_all_myerson_values()
        for my, sol in zip(my_values, solution):
            assert my == pytest.approx(sol, abs=1e-5), f"{my_values=}, {solution=}"


class TestMyersonSamplingExplainer:

    def test_with_restrict(self, regression_setup):
        model, graph, solution = regression_setup
        sampler = MyersonSamplingExplainer(graph, model, seed=42)
        my_values = sampler.sample_all_myerson_values()
        for my, sol in zip(my_values, solution):
            assert my == pytest.approx(sol, rel=1e-1, abs=1e-1), f"{my_values=}, {solution=}"

class TestMyersonClassExplainer:

    def test_with_restrict(self, classification_setup):
        model, graph, solution = classification_setup
        explainer = MyersonClassExplainer(graph, model)
        my_values = explainer.calculate_all_myerson_values()
        for my, sol in zip(my_values, solution):
            assert np.allclose(my, sol, atol=0.0001), f"{my_values=}, {solution=}"

class TestMyersonSamplingClassExplainer:

    def test_with_restrict(self, classification_setup):
        model, graph, solution = classification_setup
        sampler = MyersonSamplingClassExplainer(graph, model, seed=42)
        my_values = sampler.sample_all_myerson_values()
        for my, sol in zip(my_values, solution):
            assert my == pytest.approx(sol, rel=1e-1, abs=1e-1), f"{my_values=}, {solution=}"


class TestModelNotMutated:

    def test_explainer_preserves_training_flag(self, regression_setup):
        model, graph, _ = regression_setup
        model.train()
        assert model.training is True
        explainer = MyersonExplainer(graph, model)
        explainer.calculate_all_myerson_values()
        assert model.training is True, "Explainer silently mutated the model's training flag"

    def test_explainer_preserves_eval_flag(self, regression_setup):
        model, graph, _ = regression_setup
        model.eval()
        assert model.training is False
        explainer = MyersonExplainer(graph, model)
        explainer.calculate_all_myerson_values()
        assert model.training is False, "Explainer silently mutated the model's training flag"