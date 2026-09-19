"""The batched worth path assumes the model is a pure per-graph function.
These tests pin the guard that detects a violation instead of silently
returning corrupted values."""
import numpy as np
import pytest
import torch

from myerson.batch_guard import BatchCoupledModelError, find_batch_coupled_modules
from myerson.pyg_explain import MyersonExplainer
from .testmodels import GATConvModel


class BatchCoupledModel(GATConvModel):
    """Identical to GATConvModel but with a normalisation that mixes graphs.

    ``x.abs().mean()`` reduces over *all* nodes in the batch (it ignores the
    ``batch`` vector), exactly like PairNorm / a hand-rolled mean-pool.
    """
    def forward(self, x, edge_index, batch, *args, **kwargs):
        x = x / (x.abs().mean() + 1e-6)
        return super().forward(x, edge_index, batch, *args, **kwargs)


@pytest.fixture(scope="module")
def graph():
    with open("tests/testgraph.pt", "rb") as f:
        return torch.load(f, map_location="cpu", weights_only=False)


def _coalitions(n_nodes):
    return [(0,), (0, 1), tuple(range(n_nodes))]


def test_pure_model_passes_the_check(graph):
    model = GATConvModel(dim_in=5, dim_out=1)
    explainer = MyersonExplainer(graph, model)
    assert explainer.verify_batch_invariance(_coalitions(graph.x.shape[0])) is True
    assert explainer._batch_check_report == "passed"


def test_batch_coupled_model_is_detected(graph):
    explainer = MyersonExplainer(graph, BatchCoupledModel(dim_in=5, dim_out=1))
    assert explainer.verify_batch_invariance(_coalitions(graph.x.shape[0])) is False
    assert "not a pure per-graph function" in explainer._batch_check_report


def test_strict_policy_raises(graph):
    explainer = MyersonExplainer(graph, BatchCoupledModel(dim_in=5, dim_out=1))
    explainer.batch_check = "strict"
    with pytest.raises(BatchCoupledModelError):
        explainer.calculate_all_myerson_values()


def test_auto_policy_falls_back_and_restores_efficiency(graph, caplog):
    """With the fallback, worths are again a function of the coalition alone,
    so the Myerson values recover efficiency; with batching forced on, they do
    not."""
    model = BatchCoupledModel(dim_in=5, dim_out=1)

    safe = MyersonExplainer(graph, model)
    with caplog.at_level("WARNING"):
        values = safe.calculate_all_myerson_values()
    assert "not batch invariant" in caplog.text
    assert values.sum() == pytest.approx(
        safe.calculate_worth_of_grand_coalition(), abs=1e-4)

    unsafe = MyersonExplainer(graph, model)
    unsafe.batch_check = "off"
    corrupted = unsafe.calculate_all_myerson_values()
    assert not np.isclose(corrupted.sum(),
                          unsafe.calculate_worth_of_grand_coalition(), atol=1e-4)


def test_static_scan_flags_batchnorm_without_running_stats():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.BatchNorm1d(4, track_running_stats=False),
    )
    assert find_batch_coupled_modules(model), "BN without running stats not flagged"
    ok = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
    assert find_batch_coupled_modules(ok) == []
