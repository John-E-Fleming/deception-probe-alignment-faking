"""Activation-steering hook factory and probe-direction extraction.

Pattern: register a forward hook on a single Llama decoder layer that adds
``alpha * steering_vector`` to the layer's residual stream output at every
generation step. ``steering_vector`` is the Apollo probe direction
*unscaled back into raw activation space* (multiplied by the StandardScaler's
``scale_``), so that adding it to raw layer-output activations is equivalent
to shifting the scaler-normalised activations by ``alpha * direction`` --
which is what the probe was trained to detect.

Apollo's published detector is the source of the direction. Probe loading
delegates to the same ``LogisticRegressionDetector.load(...)`` pattern that
all our other scripts use; this module duplicates the loader's body one more
time but exposes a single canonical helper so future probe-using scripts can
import from here rather than re-inlining (the TODO across the dev scripts).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from alignment_faking_probes.probes.training import ProbeResult

if TYPE_CHECKING:
    from transformers import PreTrainedModel


#: Apollo's pinned commit hash; recorded in the ProbeResult metadata so the
#: same provenance trail other scripts maintain carries through here too.
_APOLLO_COMMIT = "f8ec4010e74927394709dffa22b97bdf8cd5a62f"


def load_probe_direction(
    apollo_example_results: Path,
    scenario: str,
    layer: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[ProbeResult, torch.Tensor]:
    """Load Apollo's published probe and return the ProbeResult + steering vector.

    The probe was trained on z-scored activations
    ``(raw - scaler_mean) / scaler_scale``, so its coefficient direction
    lives in the normalised space. To steer in *raw* residual-stream space
    (where forward hooks see activations before any normalisation), we
    multiply the direction by ``scaler_scale``. Then adding
    ``alpha * steering_vector`` to raw activations is mathematically
    equivalent to adding ``alpha * direction`` to the scaler-normalised
    activations -- which is what the probe was trained to detect.

    Args:
        apollo_example_results: Path to Apollo's ``example_results`` dir.
            Default RunPod location is
            ``/workspace/deception-detection/example_results``.
        scenario: Subdirectory name (e.g. ``"roleplaying"``, ``"followup"``,
            ``"instructed_pairs"``). Must contain a ``detector.pt`` file.
        layer: Residual-stream layer the probe was trained on (22 for the
            three Apollo probes we use).
        dtype: dtype of the returned steering vector. ``bfloat16`` matches
            the model's residual-stream dtype on a 4-bit-quantised Llama;
            change to ``float16`` or ``float32`` for other setups.

    Returns:
        ``(probe_result, steering_vector)``:
        - ``probe_result`` is a fully-populated ProbeResult mirroring what
          ``_load_apollo_published_probe`` returns in the existing dev
          scripts; useful for re-scoring activations after a steering run.
        - ``steering_vector`` is a 1-D torch tensor of shape ``(hidden_dim,)``
          on CPU. Move to ``model.device`` before registering the hook.

    Raises:
        FileNotFoundError: ``detector.pt`` is missing.
        ValueError: ``layer`` is not in the detector's trained layers.
    """
    from deception_detection.detectors import LogisticRegressionDetector

    detector_path = apollo_example_results / scenario / "detector.pt"
    if not detector_path.exists():
        raise FileNotFoundError(
            f"Apollo published probe not found at {detector_path}. "
            "Check apollo_example_results points to deception-detection/example_results."
        )
    detector = LogisticRegressionDetector.load(detector_path)
    if layer not in detector.layers:
        raise ValueError(
            f"Layer {layer} not in Apollo detector layers ({detector.layers}) "
            f"for scenario {scenario!r}."
        )
    layer_idx = detector.layers.index(layer)

    direction_np = detector.directions[layer_idx].cpu().numpy().astype(np.float64)
    scaler_mean_np = detector.scaler_mean[layer_idx].cpu().numpy().astype(np.float64)
    scaler_scale_np = detector.scaler_scale[layer_idx].cpu().numpy().astype(np.float64)

    # The unscaled steering vector. Adding alpha * this to raw layer-22
    # activations is equivalent to adding alpha * direction to the
    # normalised activations the probe was trained on.
    unscaled_np = direction_np * scaler_scale_np
    steering_vector = torch.tensor(unscaled_np, dtype=dtype)

    # Rebuild the sklearn ProbeResult to match _load_apollo_published_probe
    # in the existing dev scripts (so callers that want to re-score
    # activations post-run can use the same predict_probe path).
    coef = direction_np.reshape(1, -1)
    probe = LogisticRegression(fit_intercept=False)
    probe.coef_ = coef
    probe.intercept_ = np.zeros(1, dtype=np.float64)
    probe.classes_ = np.array([0, 1])
    probe.n_features_in_ = coef.shape[1]

    scaler = StandardScaler()
    scaler.mean_ = scaler_mean_np
    scaler.scale_ = scaler_scale_np
    scaler.var_ = scaler_scale_np**2
    scaler.n_features_in_ = coef.shape[1]

    probe_result = ProbeResult(
        probe=probe,
        scaler=scaler,
        layer=layer,
        train_scenario=f"apollo_{scenario}",
        train_auroc=float("nan"),
        n_train=0,
        n_positive=0,
        metadata={"source": "apollo_published", "apollo_commit": _APOLLO_COMMIT},
    )
    return probe_result, steering_vector


def make_steering_hook(
    alpha: float,
    steering_vector: torch.Tensor,
) -> Callable[[Any, Any, Any], Any]:
    """Build a forward hook that adds ``alpha * steering_vector`` to layer output.

    Llama decoder layers return either a tensor of shape
    ``(batch, seq_len, hidden_dim)`` OR a tuple ``(hidden_states, ...)``
    depending on call kwargs. The hook handles both forms, modifies the
    hidden_states in place by adding ``alpha * steering_vector`` (broadcast
    over batch and sequence), and returns the same shape it received.

    Args:
        alpha: Steering strength. ``0`` is a no-op (returns input unchanged).
        steering_vector: 1-D tensor of shape ``(hidden_dim,)``. Must already
            be on the same device as the layer's output. The hook will cast
            it to the output's dtype if needed.

    Returns:
        A callable suitable for ``module.register_forward_hook(...)``.
        Signature follows PyTorch convention: ``(module, input, output)``.
    """

    def hook(_module: Any, _input: Any, output: Any) -> Any:
        if alpha == 0.0:
            return output
        # Llama decoder layer output is normally a tuple
        # (hidden_states, present_key_value, ...). The hidden_states are
        # at position 0.
        if isinstance(output, tuple):
            hidden_states = output[0]
            shifted = _apply_steering(hidden_states, alpha, steering_vector)
            return (shifted, *output[1:])
        # Fall through: some configurations return just the tensor.
        return _apply_steering(output, alpha, steering_vector)

    return hook


def _apply_steering(
    hidden_states: torch.Tensor,
    alpha: float,
    steering_vector: torch.Tensor,
) -> torch.Tensor:
    """Add ``alpha * steering_vector`` to hidden_states, broadcasting over batch/seq.

    ``hidden_states`` has shape ``(batch, seq_len, hidden_dim)`` or
    ``(batch, hidden_dim)`` depending on whether the layer was called with
    a sequence. ``steering_vector`` is 1-D and broadcasts to the trailing
    dim. dtype is matched to hidden_states.
    """
    vec = steering_vector.to(hidden_states.device, dtype=hidden_states.dtype)
    return hidden_states + alpha * vec


@contextmanager
def steering_active(
    model: PreTrainedModel,
    layer_idx: int,
    alpha: float,
    steering_vector: torch.Tensor,
) -> Iterator[None]:
    """Register a steering hook on entry; remove it on exit (incl. exceptions).

    Targets ``model.model.layers[layer_idx]`` on a Llama model. PeftModel
    wrappers forward the ``model.model.layers`` attribute access through to
    the base model, so this works for both the bare base model and the
    PEFT-wrapped LoRA-adapted version.

    Args:
        model: The HuggingFace model (or PeftModel) whose layer we'll hook.
        layer_idx: Which decoder layer to attach to. For our experiments this
            is always 22 (matching the probe's training layer).
        alpha: Steering strength; ``0`` is a no-op via ``make_steering_hook``.
        steering_vector: 1-D tensor on CPU or model.device. The hook moves
            it to the output's device on first invocation.

    Yields:
        ``None``. Caller's body runs with the hook installed; on exit the
        hook is removed via the handle returned by ``register_forward_hook``.

    Raises:
        AttributeError: If ``model.model.layers[layer_idx]`` doesn't resolve.
            The PeftModel attribute chain is ``model.base_model.model``, so
            on PeftModel, ``model.model.layers`` follows the wrapper's
            __getattr__ fallback. If that ever breaks in a future PEFT
            release, the fix is to walk ``model.base_model.model.model.layers``
            explicitly.
    """
    target_layer = _resolve_decoder_layer(model, layer_idx)
    hook = make_steering_hook(alpha=alpha, steering_vector=steering_vector)
    handle = target_layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def _resolve_decoder_layer(model: PreTrainedModel, layer_idx: int) -> torch.nn.Module:
    """Walk the model's attribute chain to find ``layers[layer_idx]``.

    Handles three common cases:
    1. Bare HF model: ``model.model.layers[i]``
    2. PeftModel wrapper: ``model.base_model.model.model.layers[i]`` (the
       inner ``model.model.layers`` is forwarded by PeftModel's __getattr__
       so case 1 usually works; fall back to the explicit chain if not).
    3. Some quantised wrappers add another indirection.
    """
    # Try the simple path first -- works for bare HF models AND for
    # PeftModel because the wrapper falls back to the base model's attrs.
    candidate = getattr(model, "model", None)
    if candidate is not None:
        layers = getattr(candidate, "layers", None)
        if layers is not None and 0 <= layer_idx < len(layers):
            return layers[layer_idx]

    # Explicit PeftModel chain: model.base_model.model.model.layers
    base = getattr(model, "base_model", None)
    if base is not None:
        inner = getattr(base, "model", None)
        if inner is not None:
            inner_inner = getattr(inner, "model", None)
            if inner_inner is not None:
                layers = getattr(inner_inner, "layers", None)
                if layers is not None and 0 <= layer_idx < len(layers):
                    return layers[layer_idx]

    raise AttributeError(
        f"Could not resolve decoder layer {layer_idx} on model of type "
        f"{type(model).__name__}. Expected model.model.layers or "
        f"model.base_model.model.model.layers."
    )
