# tests/conftest.py
"""Backend gating for the optional PyG / chemprop explainer suites.

The core game-theory tests are pure Python; the two GNN backends are optional
extras. Modules that import a missing backend are dropped at collection time
(an import guard inside them would be too late -- their helper imports, e.g.
``testmodels``, pull in the backend themselves).
"""
from importlib.util import find_spec

import pytest


def _available(*modules) -> bool:
    try:
        return all(find_spec(m) is not None for m in modules)
    except (ImportError, ValueError):
        return False


HAS_PYG = _available("torch", "torch_geometric")
HAS_CHEMPROP = _available("chemprop")

# Whole modules are skipped, not individual tests: the backend is needed to
# import them at all.
collect_ignore = []
if not HAS_PYG:
    collect_ignore += [
        "test_myerson_explain.py",
        "test_perturbation_explain.py",
        "test_shapley_explain.py",
        "test_batch_guard.py",
        "testmodels.py",
        "test_disconnected_warning.py",
    ]
if not HAS_CHEMPROP:
    collect_ignore += ["test_myerson_mpnn_explain.py"]


# Auto-apply backend markers by filename, so no test file needs `pytestmark`.
_MARKS = {
    "test_myerson_explain.py": ("pyg", "slow"),
    "test_perturbation_explain.py": ("pyg", "slow"),
    "test_shapley_explain.py": ("pyg", "slow"),
    "test_myerson_mpnn_explain.py": ("chemprop", "slow"),
}


def pytest_collection_modifyitems(items):
    for item in items:
        for mark in _MARKS.get(item.path.name, ()):
            item.add_marker(getattr(pytest.mark, mark))