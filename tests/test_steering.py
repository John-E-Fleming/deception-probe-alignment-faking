"""Tests for the activation-steering hook + helpers.

These cover the pure-torch parts of ``alignment_faking_probes.steering.hooks``:
- the forward-hook factory (no-op at alpha=0, additive at alpha != 0, tuple
  vs tensor output handling)
- the steering_active context manager (registers + removes the hook;
  removal still happens on exception)
- decoder-layer resolution across model attribute layouts

``load_probe_direction`` is NOT tested here -- it depends on Apollo's
``deception_detection.detectors.LogisticRegressionDetector`` which is
Linux-only and pod-only. Coverage of that function is exercised at runtime
via ``scripts/08_run_steering_experiment.py``.
"""

from __future__ import annotations

import pytest
import torch

from alignment_faking_probes.steering.hooks import (
    _resolve_decoder_layer,
    make_steering_hook,
    steering_active,
)

# --- make_steering_hook --------------------------------------------------


def test_steering_hook_is_noop_at_alpha_zero() -> None:
    vec = torch.tensor([1.0, 2.0, 3.0])
    hook = make_steering_hook(alpha=0.0, steering_vector=vec)
    hidden = torch.zeros(2, 4, 3)  # (batch, seq, hidden)
    output = hook(None, None, hidden)
    assert torch.equal(output, hidden)


def test_steering_hook_adds_alpha_times_vector_to_tensor_output() -> None:
    vec = torch.tensor([1.0, 2.0, 3.0])
    hook = make_steering_hook(alpha=2.0, steering_vector=vec)
    hidden = torch.zeros(2, 4, 3)
    output = hook(None, None, hidden)
    # Every (batch, seq) position should now equal alpha * vec
    expected = torch.zeros(2, 4, 3) + torch.tensor([2.0, 4.0, 6.0])
    assert torch.allclose(output, expected)


def test_steering_hook_preserves_tuple_output_structure() -> None:
    """Llama decoder layers return (hidden_states, present_kv, ...);
    the hook must rewrap, not flatten to a bare tensor."""
    vec = torch.tensor([1.0, 0.0, 0.0])
    hook = make_steering_hook(alpha=1.0, steering_vector=vec)
    hidden = torch.zeros(1, 2, 3)
    aux = torch.ones(5)  # something else the layer returned
    output = hook(None, None, (hidden, aux))
    assert isinstance(output, tuple)
    assert len(output) == 2
    expected_hidden = torch.zeros(1, 2, 3) + torch.tensor([1.0, 0.0, 0.0])
    assert torch.allclose(output[0], expected_hidden)
    # Aux tensor must pass through untouched (same identity, not just equal)
    assert output[1] is aux


def test_steering_hook_casts_vector_dtype_to_hidden_states() -> None:
    """Steering vector might be bfloat16 on disk but the hidden states
    could come through in float32 if downstream code requested it. Hook
    should cast to the hidden_states dtype, not the other way around."""
    vec = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
    hook = make_steering_hook(alpha=1.0, steering_vector=vec)
    hidden = torch.zeros(1, 2, 3, dtype=torch.float32)
    output = hook(None, None, hidden)
    assert output.dtype == torch.float32


def test_steering_hook_broadcasts_over_batch_and_sequence() -> None:
    vec = torch.tensor([10.0, 20.0, 30.0])
    hook = make_steering_hook(alpha=0.5, steering_vector=vec)
    hidden = torch.zeros(3, 7, 3)  # batch=3, seq=7
    output = hook(None, None, hidden)
    expected_per_pos = torch.tensor([5.0, 10.0, 15.0])
    # Every position should have the same shift
    for b in range(3):
        for s in range(7):
            assert torch.equal(output[b, s], expected_per_pos)


# --- _resolve_decoder_layer ---------------------------------------------


class _MockLayer:
    """Stand-in for a transformers decoder layer."""

    def __init__(self, idx: int) -> None:
        self.idx = idx


class _MockInner:
    def __init__(self, n_layers: int = 4) -> None:
        self.layers = [_MockLayer(i) for i in range(n_layers)]


class _MockModel:
    """Bare-HF-model layout: model.model.layers[i]."""

    def __init__(self) -> None:
        self.model = _MockInner()


class _MockPeftModel:
    """PeftModel layout: model.base_model.model.model.layers[i].

    The real PeftModel also forwards .model via __getattr__, so the simple
    path works too -- but our resolver should handle the explicit chain as
    a fallback.
    """

    def __init__(self) -> None:
        self.base_model = _MockBase()


class _MockBase:
    def __init__(self) -> None:
        self.model = _MockModel()


def test_resolve_decoder_layer_handles_bare_hf_model() -> None:
    model = _MockModel()
    layer = _resolve_decoder_layer(model, layer_idx=2)
    assert isinstance(layer, _MockLayer)
    assert layer.idx == 2


def test_resolve_decoder_layer_handles_peft_wrapper() -> None:
    """When the simple path doesn't work, the explicit base_model chain
    should resolve the layer."""

    # A model where `.model` exists but doesn't have `.layers` -- forces
    # the resolver to walk the explicit base_model.model.model.layers chain.
    class _PeftWithoutForwarding:
        def __init__(self) -> None:
            self.model = _ShellWithoutLayers()
            self.base_model = _MockBase()

    class _ShellWithoutLayers:
        pass  # no .layers attribute

    model = _PeftWithoutForwarding()
    layer = _resolve_decoder_layer(model, layer_idx=1)
    assert isinstance(layer, _MockLayer)
    assert layer.idx == 1


def test_resolve_decoder_layer_raises_when_chain_missing() -> None:
    class _NoLayers:
        pass

    with pytest.raises(AttributeError, match="Could not resolve decoder layer"):
        _resolve_decoder_layer(_NoLayers(), layer_idx=0)


def test_resolve_decoder_layer_raises_when_index_out_of_range() -> None:
    model = _MockModel()  # 4 layers
    with pytest.raises(AttributeError, match="Could not resolve decoder layer"):
        _resolve_decoder_layer(model, layer_idx=99)


# --- steering_active context manager -------------------------------------


class _HookableLayer(torch.nn.Module):
    """A trivial nn.Module so we can register / remove real forward hooks."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x  # identity


class _ModelWithLayers:
    """Bare-HF-style model with .model.layers[i] populated by _HookableLayers."""

    def __init__(self, n_layers: int, hidden_dim: int) -> None:
        self.model = _LayerContainer(n_layers, hidden_dim)


class _LayerContainer:
    def __init__(self, n_layers: int, hidden_dim: int) -> None:
        self.layers = torch.nn.ModuleList([_HookableLayer(hidden_dim) for _ in range(n_layers)])


def test_steering_active_attaches_and_removes_hook() -> None:
    model = _ModelWithLayers(n_layers=4, hidden_dim=3)
    vec = torch.tensor([1.0, 2.0, 3.0])
    layer = model.model.layers[2]

    # Before entering: no hooks
    assert len(layer._forward_hooks) == 0

    with steering_active(model, layer_idx=2, alpha=1.0, steering_vector=vec):
        # During: exactly one hook
        assert len(layer._forward_hooks) == 1

    # After: hook removed
    assert len(layer._forward_hooks) == 0


def test_steering_active_removes_hook_on_exception() -> None:
    model = _ModelWithLayers(n_layers=4, hidden_dim=3)
    vec = torch.tensor([1.0, 2.0, 3.0])
    layer = model.model.layers[1]

    with pytest.raises(RuntimeError, match="boom"):
        with steering_active(model, layer_idx=1, alpha=1.0, steering_vector=vec):
            assert len(layer._forward_hooks) == 1
            raise RuntimeError("boom")

    # Hook was removed even though the body raised
    assert len(layer._forward_hooks) == 0


def test_steering_active_actually_modifies_layer_output() -> None:
    """End-to-end check: register the hook via the context manager, run a
    forward pass, confirm the output is shifted by alpha * vector."""
    model = _ModelWithLayers(n_layers=2, hidden_dim=3)
    vec = torch.tensor([1.0, 2.0, 3.0])
    layer = model.model.layers[0]

    test_input = torch.zeros(1, 5, 3)  # batch=1, seq=5, hidden=3

    # Baseline -- no hook, identity layer returns zeros
    baseline = layer(test_input)
    assert torch.equal(baseline, torch.zeros(1, 5, 3))

    # With steering active at alpha=2.0
    with steering_active(model, layer_idx=0, alpha=2.0, steering_vector=vec):
        steered = layer(test_input)
    expected = torch.zeros(1, 5, 3) + torch.tensor([2.0, 4.0, 6.0])
    assert torch.allclose(steered, expected)

    # And the hook is gone -- another forward returns identity again
    after = layer(test_input)
    assert torch.equal(after, torch.zeros(1, 5, 3))
