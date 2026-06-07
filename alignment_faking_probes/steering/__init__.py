"""Activation-steering primitives for causal-intervention experiments.

Exposes the public surface needed by ``scripts/08_run_steering_experiment.py``:

- ``load_probe_direction``: load Apollo's published detector and return BOTH
  the ProbeResult (for re-scoring) AND an unscaled steering vector ready to
  add to raw residual-stream activations.
- ``make_steering_hook``: factory for a forward hook that adds
  ``alpha * steering_vector`` to a LlamaDecoderLayer's output.
- ``steering_active``: context manager that registers the hook on entry and
  removes it on exit (including exception paths).

The Apollo-coupled probe-loading import is lazy inside ``load_probe_direction``
so the module is importable locally without Apollo installed; the actual
steering hook + context manager are pure-torch and unit-testable.
"""

from alignment_faking_probes.steering.hooks import (
    load_probe_direction,
    make_steering_hook,
    steering_active,
)

__all__ = ["load_probe_direction", "make_steering_hook", "steering_active"]
