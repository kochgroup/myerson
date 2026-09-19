"""The batched worth path assumes the model is a pure per-graph function.
These tests pin the guard that detects a violation instead of silently
returning corrupted values.

Everything here is deterministic by construction: the models are seeded and
every assertion is expressed *relative to the magnitude of the worths*. A fixed
absolute tolerance on a model output whose scale depends on a random weight
draw makes the test a coin flip, not a regression detector.
"""
import numpy as np
import pytest
import torch

from myerson.batch_guard import BatchCoupledModelError, find_batch_coupled_modules
from myerson.pyg_explain import MyersonExplainer
from .testmodels import GATConvModel

SEED = 0


class BatchCoupledModel(GATConvModel):
    """GAT with a *size-dependent readout*: it divides by the number of nodes in
    the whole forward pass instead of per graph (the hand-rolled
    ``global_mean_pool`` bug listed in ``batch_guard``'s docstring).

    For a single subgraph the divisor is ``|S|``; inside a batch it is the total
    node count of the chunk, so ``v(S)`` depends on which other subgraphs happen
    to share its batch. The error is *multiplicative*, of order
    ``chunk_nodes / |S|``, i.e. orders of magnitude above float noise for any
    weight draw -- unlike an additive perturbation of the input features, it
    cannot be damped into the noise floor by an unlucky initialisation.
    """

    def forward(self, x, edge_index, batch, *args, **kwargs):
        out = super().forward(x, edge_index, batch, *args, **kwargs)
        return out / x.shape[0]          # should be global_mean_pool(x, batch)


class TinyOutputModel(GATConvModel):
    """Pure per-graph model whose outputs are ~1e-9.

    Regression guard for the mirror-image failure: a *fixed absolute* tolerance
    either drowns real coupling or flags float noise as soon as the output scale
    drifts. The verdict must not depend on the magnitude of the prediction.
    """

    def forward(self, x, edge_index, batch, *args, **kwargs):
        return super().forward(x, edge_index, batch, *args, **kwargs) * 1e-9


def _model(cls, dim_out: int = 1):
    """Seeded model construction -- the whole point of this rewrite."""
    torch.manual_seed(SEED)
    return cls(dim_in=5, dim_out=dim_out)


@pytest.fixture(scope="module")
def graph():
    with open("tests/testgraph.pt", "rb") as f:
        return torch.load(f, map_location="cpu", weights_only=False)


def _probes(n_nodes):
    """Size-heterogeneous coalitions: coupling only shows on mixed batches."""
    return [(0,), (0, 1, 2), tuple(range(n_nodes))]


def _scale(values, grand) -> float:
    """Reference magnitude for scale-free assertions."""
    return max(abs(float(grand)), float(np.abs(values).max()))


def test_pure_model_passes_the_check(graph):
    explainer = MyersonExplainer(graph, _model(GATConvModel))
    assert explainer.verify_batch_invariance(_probes(graph.x.shape[0])) is True
    assert explainer._batch_check_report.startswith("passed")


def test_pure_model_with_tiny_outputs_is_not_a_false_positive(graph):
    """A 1e-9-scale prediction is still batch invariant; the guard must say so
    without leaning on a hard-coded absolute tolerance."""
    explainer = MyersonExplainer(graph, _model(TinyOutputModel))
    assert explainer.verify_batch_invariance(_probes(graph.x.shape[0])) is True


def test_batch_coupled_model_is_detected(graph):
    explainer = MyersonExplainer(graph, _model(BatchCoupledModel))
    assert explainer.verify_batch_invariance(_probes(graph.x.shape[0])) is False
    assert "not a pure per-graph function" in explainer._batch_check_report


def test_strict_policy_raises(graph):
    explainer = MyersonExplainer(graph, _model(BatchCoupledModel))
    explainer.batch_check = "strict"
    with pytest.raises(BatchCoupledModelError):
        explainer.calculate_all_myerson_values()


def test_auto_policy_warns_and_restores_efficiency(graph, caplog):
    """The fallback makes ``v(S)`` a function of ``S`` alone again, so the
    Myerson values recover efficiency to float precision."""
    safe = MyersonExplainer(graph, _model(BatchCoupledModel))  # batch_check="auto"
    with caplog.at_level("WARNING"):
        values = safe.calculate_all_myerson_values()
    assert "not batch invariant" in caplog.text

    grand = safe.calculate_worth_of_grand_coalition()
    scale = _scale(values, grand)
    assert scale > 0.0, "degenerate model output would make the test vacuous"
    assert abs(values.sum() - grand) < 1e-4 * scale


def test_disabling_the_guard_reproduces_the_corruption(graph):
    """Opting out must demonstrably break efficiency -- by a large *relative*
    amount (~ 1 - |N| / chunk_nodes here), never by a hard-coded epsilon."""
    guarded = MyersonExplainer(graph, _model(BatchCoupledModel))
    values = guarded.calculate_all_myerson_values()
    grand = guarded.calculate_worth_of_grand_coalition()
    scale = _scale(values, grand)

    unsafe = MyersonExplainer(graph, _model(BatchCoupledModel))
    unsafe.batch_check = "off"
    corrupted = unsafe.calculate_all_myerson_values()

    assert abs(corrupted.sum() - grand) > 0.1 * scale
    assert not np.allclose(corrupted, values, rtol=1e-3, atol=0.0)


def test_static_scan_flags_batchnorm_without_running_stats():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.BatchNorm1d(4, track_running_stats=False),
    )
    assert find_batch_coupled_modules(model), "BN without running stats not flagged"
    ok = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
    assert find_batch_coupled_modules(ok) == []
