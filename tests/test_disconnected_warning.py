"""A disconnected input must warn: prediction != sum of Myerson values."""
import logging

import pytest
import torch
from torch_geometric.data import Data

from myerson.pyg_explain import MyersonExplainer, MyersonClassExplainer
from .testmodels import GATConvModel

def _graph(edges, n=6):
    torch.manual_seed(0)
    both = [e for (u, v) in edges for e in ((u, v), (v, u))]
    return Data(x=torch.randn(n, 5),
                edge_index=torch.tensor(both).t().contiguous())

DISCONNECTED = [(0, 1), (1, 2), (3, 4)]          # + isolated node 5 => 3 components
CONNECTED = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

@pytest.mark.parametrize("cls,dim_out", [(MyersonExplainer, 1),
                                         (MyersonClassExplainer, 3)])
def test_disconnected_graph_warns(cls, dim_out, caplog):
    """Both the scalar and the multi-task explainer warn, and rendering the
    INFO diagnostic does not blow up on a per-task vector prediction."""
    torch.manual_seed(0)
    with caplog.at_level(logging.INFO):
        cls(_graph(DISCONNECTED), GATConvModel(dim_in=5, dim_out=dim_out))
    messages = [r.getMessage() for r in caplog.records]
    assert any("3 individual components" in m for m in messages), messages
    assert any("Prediction=" in m and "Worth=" in m for m in messages), messages

def test_connected_graph_does_not_warn(caplog):
    """Negative control, so the test above can't pass for the wrong reason."""
    torch.manual_seed(0)
    with caplog.at_level(logging.INFO):
        MyersonExplainer(_graph(CONNECTED), GATConvModel(dim_in=5, dim_out=1))
    assert not [r for r in caplog.records if "individual components" in r.getMessage()]
