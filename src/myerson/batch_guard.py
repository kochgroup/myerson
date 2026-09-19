"""Safety guards for the *batched* worth evaluation used by the explainers.

Every batched code path in this package (``_batch_data_from_coalitions``,
``_batch_mol_graph_from_coalitions``) collates many connected subgraphs into a
single disjoint-union graph and runs **one** forward pass. That is only valid if
the coalition function is a pure per-graph function::

    model(disjoint_union(G_1, ..., G_B))[i] == model(G_i)      # batch invariance

Ops that couple samples inside a batch break this silently:

* ``BatchNorm`` with ``track_running_stats=False`` (batch statistics are used
  even in ``eval()`` mode), ``SyncBatchNorm``;
* PyG ``PairNorm`` / ``MeanSubtractionNorm`` and any custom normalisation that
  reduces over the node dimension without consuming the ``batch`` vector;
* size-dependent readouts (e.g. dividing by ``x.shape[0]`` instead of using
  ``global_mean_pool(x, batch)``);
* non-deterministic forwards (dropout that stays active, stochastic depth).

The guard here is *empirical*: it compares the batched worth against the
single-subgraph worth for a few real coalitions. The static module scan
(:func:`find_batch_coupled_modules`) is only used to name a likely culprit in
the diagnostic message.

Custom modules can opt in to the static scan by setting
``module._myerson_batch_coupled = True``.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("myerson.batch_guard")

__all__ = [
    "BatchCoupledModelError",
    "BatchInvarianceGuardMixin",
    "find_batch_coupled_modules",
]


class BatchCoupledModelError(RuntimeError):
    """Raised when the coalition function is not a pure per-graph function.

    Signals that batched worth evaluation would corrupt the values. Either fix
    the model (feed ``batch`` to the normalisation / use running statistics) or
    set ``explainer.batch_check = 'auto'`` to fall back to one forward pass per
    connected subgraph.
    """


_ALWAYS_COUPLED = frozenset({
    "PairNorm",
    "MeanSubtractionNorm",
})

_BATCHNORM_TYPES = frozenset({
    "BatchNorm",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "SyncBatchNorm",
    "HeteroBatchNorm",
})


def find_batch_coupled_modules(model) -> list[str]:
    """Statically list submodules that are known to couple samples in a batch.

    Purely diagnostic (heuristic, may miss functional ops written inline in
    ``forward``); the authoritative check is
    :meth:`BatchInvarianceGuardMixin.verify_batch_invariance`.

    Args:
        model: Any object exposing ``named_modules`` (a ``torch.nn.Module``).

    Returns:
        list[str]: ``"<qualified name> (<class name>)"`` for each suspect.
    """
    named_modules = getattr(model, "named_modules", None)
    if named_modules is None:
        return []

    hits: list[str] = []
    seen: set[int] = set()
    for name, module in named_modules():
        if id(module) in seen:
            continue
        seen.add(id(module))
        cls = type(module).__name__
        label = f"{name or '<root>'} ({cls})"
        if getattr(module, "_myerson_batch_coupled", False):
            hits.append(label)
        elif cls in _ALWAYS_COUPLED:
            hits.append(label)
        elif cls in _BATCHNORM_TYPES:
            inner = getattr(module, "module", module)
            if (not getattr(inner, "track_running_stats", True)
                    or getattr(inner, "running_mean", None) is None):
                hits.append(label + " [track_running_stats=False]")
    return hits


class BatchInvarianceGuardMixin:
    """Mixin adding a one-off batch-invariance check to a batched explainer.

    The concrete explainer must provide two thin hooks:

        * :meth:`_single_worth` — worth of one connected subgraph, computed via
          the *single-subgraph* path (one forward, batch of one graph built
          independently of the vectorised collator).
        * :meth:`_worth_of_chunk` — worths of a list of connected subgraphs,
          computed via the *batched* path (one forward for the whole chunk).

    and call :meth:`_guard_batch_size` once, right before its chunking loop.
    """

    batch_check: str = "auto"              # "auto" | "strict" | "off"
    #: Relative tolerance, applied to the magnitude of the probe worths.
    batch_check_rtol: float = 1e-3
    #: Hard absolute floor. Deliberately tiny: the operative tolerance is
    #: derived from the *measured* noise floor, not from a constant guessed at
    #: authoring time (a constant either masks coupling in small-output models
    #: or flags noise in models with heavy cancellation).
    batch_check_atol: float = 1e-9
    #: Safety factor applied to the measured float-noise floor.
    batch_check_noise_factor: float = 16.0
    batch_check_max_probes: int = 3

    _batch_safe: bool | None = None        # cached verdict (per instance)
    _batch_check_report: str = ""          # human-readable detail of the verdict
    _batch_check_tol: float = 0.0           # tolerance actually used (diagnostics)

    def _single_worth(self, coalition):
        """Worth of ``coalition`` via the single-subgraph (unbatched) path."""
        raise NotImplementedError

    def _worth_of_chunk(self, coalitions: list) -> list:
        """Worths of ``coalitions`` via the batched path, in input order."""
        raise NotImplementedError

    @staticmethod
    def _as_float_array(value) -> np.ndarray:
        """Flatten a scalar / array / (inference-mode) torch tensor worth."""
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        try:
            return np.asarray(value, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return np.asarray(value.tolist(), dtype=np.float64).reshape(-1)

    @staticmethod
    def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
        """Largest absolute difference, ``inf`` on a shape mismatch."""
        if a.shape != b.shape:
            return float("inf")
        if a.size == 0:
            return 0.0
        return float(np.abs(a - b).max())

    def _probe_coalitions(self, coalitions) -> list[tuple]:
        """Pick a few *size-heterogeneous* probes from the real workload."""
        unique = {tuple(c) for c in coalitions if c}
        if len(unique) < 2:
            return []
        ordered = sorted(unique, key=lambda c: (len(c), c))
        k = max(2, min(int(self.batch_check_max_probes), len(ordered)))
        idx = sorted({round(i * (len(ordered) - 1) / (k - 1)) for i in range(k)})
        return [ordered[i] for i in idx]

    def verify_batch_invariance(self, coalitions) -> bool:
        """Probe the model: does batching change the worths?

        Four checks, all on *real* coalitions from the current workload:

        1. **determinism** - two identical single-graph forwards must agree;
        2. **noise floor** - the batched collator on a batch of *one* graph must
           reproduce the single-subgraph path. A batch of one cannot couple
           anything, so whatever difference remains is the float
           reduction-order noise of the two code paths. This *calibrates* the
           tolerance instead of hard-coding one, which is what keeps the verdict
           independent of the model's output scale;
        3. **shape** - batched and single worths must agree in shape;
        4. **invariance** - one batched forward over size-heterogeneous probes
           must reproduce the per-probe single forwards within that tolerance.

        Costs ``max_probes + 3`` forwards, once per instance.

        Returns:
            bool: ``True`` if batching is safe. Details (including the
            tolerance used) are left in ``self._batch_check_report``.
        """
        probes = self._probe_coalitions(coalitions)
        if len(probes) < 2:
            self._batch_check_report = (
                "skipped (fewer than two distinct connected subgraphs)")
            return True

        # Calibrate on the *largest* probe: most reduction ops, most noise.
        pivot = probes[-1]
        first = self._as_float_array(self._single_worth(pivot))
        again = self._as_float_array(self._single_worth(pivot))
        solo = self._as_float_array(self._worth_of_chunk([pivot])[0])

        singles = [self._as_float_array(self._single_worth(c))
                   if c != pivot else first for c in probes]
        scale = max((float(np.abs(s).max()) for s in singles if s.size),
                    default=0.0)

        repeat_noise = self._max_abs_diff(first, again)
        if repeat_noise > self.batch_check_rtol * scale + self.batch_check_atol:
            self._batch_check_report = (
                "the model is non-deterministic in eval mode (two identical "
                f"single-graph forwards differ by {repeat_noise:.3g}); batched "
                "and unbatched worths cannot be reconciled")
            return False

        # A batch of one cannot couple graphs -> this is pure float noise.
        collator_noise = self._max_abs_diff(first, solo)
        if collator_noise == float("inf"):
            self._batch_check_report = (
                f"batched worth has shape {solo.shape} but the single-subgraph "
                f"worth has shape {first.shape} for coalition {pivot}")
            return False

        noise = max(repeat_noise, collator_noise)
        tol = (self.batch_check_atol
               + self.batch_check_rtol * scale
               + self.batch_check_noise_factor * noise)
        self._batch_check_tol = tol

        batched = [self._as_float_array(w)
                   for w in self._worth_of_chunk(list(probes))]

        worst, worst_coalition = 0.0, None
        for coalition, b, s in zip(probes, batched, singles, strict=True):
            diff = self._max_abs_diff(b, s)
            if diff == float("inf"):
                self._batch_check_report = (
                    f"batched worth has shape {b.shape} but the single-subgraph "
                    f"worth has shape {s.shape} for coalition {coalition}")
                return False
            if diff > tol and diff > worst:
                worst, worst_coalition = diff, coalition

        if worst_coalition is None:
            self._batch_check_report = (
                f"passed (max deviation <= tol={tol:.3g}, "
                f"noise floor {noise:.3g}, worth scale {scale:.3g})")
            return True

        suspects = find_batch_coupled_modules(
            getattr(self, "coalition_function", None))
        self._batch_check_report = (
            f"batching changes the worth by up to {worst:.3g} (coalition "
            f"{worst_coalition}, tol={tol:.3g}, measured float-noise floor "
            f"{noise:.3g}); the model is not a pure per-graph function"
            + (f"; suspect modules: {', '.join(suspects)}" if suspects else
               "; no batch-coupled module found by the static scan, check for "
               "size-dependent ops written inline in forward()"))
        return False

    def _guard_batch_size(self, coalitions, batch_size: int) -> int:
        """Validate the model once and return a safe ``batch_size``."""
        policy = str(getattr(self, "batch_check", "auto")).lower()
        if policy not in ("auto", "strict", "off"):
            raise ValueError(
                f"batch_check must be 'auto', 'strict' or 'off', got {policy!r}")
        if policy == "off" or batch_size <= 1:
            return batch_size

        if self._batch_safe is None:
            self._batch_safe = self.verify_batch_invariance(coalitions)
        if self._batch_safe:
            return batch_size

        logger = getattr(self, "log", log)
        message = ("Coalition function is not batch invariant: "
                   f"{self._batch_check_report}. Batched worth evaluation would "
                   "silently corrupt the values.")
        if policy == "strict":
            raise BatchCoupledModelError(message)
        logger.warning(
            "%s Falling back to batch_size=1 (one forward per connected "
            "subgraph, much slower). Set <explainer>.batch_check='off' to keep "
            "batching anyway, or 'strict' to raise instead.", message)
        return 1
