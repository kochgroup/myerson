"""A python package to calculate Myerson values from game theory and use them as explanations for graph neural networks."""

from .myerson import MyersonCalculator, MyersonSampler, MyersonBudgetExceeded
from .shapley import ShapleyCalculator, ShapleySampler
from .batch_guard import BatchCoupledModelError, BatchInvarianceGuardMixin, find_batch_coupled_modules
import logging

log = logging.getLogger(__name__)

__all__ = [
    "MyersonCalculator",
    "MyersonSampler",
    "MyersonBudgetExceeded",
    "ShapleyCalculator",
    "ShapleySampler",
    "BatchCoupledModelError",
    "BatchInvarianceGuardMixin",
    "find_batch_coupled_modules",
    "pyg_explain",
    "chemprop_explain",
]

__version__ = "1.0.1" # update this
