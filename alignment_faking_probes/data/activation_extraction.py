"""Extract mean-pooled residual stream activations via NNSight.

This module hooks into specified residual-stream layers of a Hugging Face
causal LM, runs a forward pass on a batch of texts, and mean-pools the
captured hidden states over the sequence dimension.

Local-testable pieces (``ExtractionConfig`` validation, ``_pool_hidden``,
``get_af_extraction_texts``) have unit tests. ``extract_activations``
itself requires a GPU and a Llama-3.3 model; its only validation is the
``@pytest.mark.gpu`` smoke test, run manually on RunPod.

**Apollo compatibility check (REQUIRED before relying on output).** Apollo's
probe weights were trained against activations produced by *their*
extraction code. If our NNSight extraction differs in pooling, layer
indexing, or normalisation, every transfer-matrix cell silently uses the
wrong numbers. Before running the real transfer experiment on RunPod,
cross-check this module's output against ``deception_detection/experiment.py``
on a single Apollo scenario and confirm activations agree to floating-point
tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from alignment_faking_probes.data.labeling import LabelledTranscript

_VALID_QUANTISATIONS = ("4bit", "8bit", "none")


@dataclass
class ExtractionConfig:
    """Configuration for residual-stream activation extraction.

    Attributes:
        model_name: Hugging Face model identifier
            (e.g. ``"meta-llama/Llama-3.3-70B-Instruct"``).
        layers: Residual stream layer indices to extract from.
        batch_size: Number of texts processed per forward pass.
        quantisation: One of ``"4bit"``, ``"8bit"``, ``"none"``. 4-bit uses
            bitsandbytes NF4. Quantisation can shift activation geometry —
            keep this consistent across all extraction runs in a project.
        device: PyTorch device specifier (``"cuda"`` or ``"cpu"``).
        max_length: Maximum token length; longer inputs are truncated. Must
            cover the longest scratchpad + response in the dataset.

    Raises:
        ValueError: If ``quantisation`` is not recognised, ``batch_size`` /
            ``max_length`` are non-positive, or ``layers`` is empty / contains
            negative indices.
    """

    model_name: str
    layers: list[int]
    batch_size: int
    quantisation: str
    device: str
    max_length: int

    def __post_init__(self) -> None:
        if self.quantisation not in _VALID_QUANTISATIONS:
            raise ValueError(
                f"quantisation must be one of {_VALID_QUANTISATIONS}, got {self.quantisation!r}"
            )
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")
        if not self.layers:
            raise ValueError("layers must be non-empty")
        if any(layer < 0 for layer in self.layers):
            raise ValueError(f"All layers must be non-negative, got {self.layers}")


def _pool_hidden(hidden: torch.Tensor, layer: int) -> np.ndarray:
    """Mean-pool a 3D hidden state tensor over the sequence dimension.

    Args:
        hidden: Tensor of shape ``(batch, seq_len, hidden_dim)``.
        layer: Layer index, used only for error messages.

    Returns:
        ndarray of shape ``(batch, hidden_dim)``.

    Raises:
        RuntimeError: If ``hidden`` is not 3D. Shape mismatches usually
            mean an upstream NNSight hook captured the wrong tensor.
    """
    if hidden.ndim != 3:
        raise RuntimeError(
            f"Layer {layer}: expected 3D tensor (batch, seq_len, hidden_dim), "
            f"got shape {tuple(hidden.shape)}"
        )
    # ``.detach()`` drops the autograd graph that nnsight's hooks attach
    # (no gradients are needed at extraction time). ``.float()`` casts
    # bf16 / fp16 activations to fp32 so numpy can represent them — numpy
    # has no bf16 dtype, and downstream probe training expects fp32.
    return hidden.detach().float().mean(dim=1).cpu().numpy()


def get_af_extraction_texts(labelled: list[LabelledTranscript]) -> list[str]:
    """Build extraction texts from AF transcripts.

    Concatenates the scratchpad and the response with a blank-line
    separator. We extract from what the model produced (its scratchpad
    reasoning + final response), not from the prompt it was given — the
    probe reads the model's internal state during its own generation.

    Args:
        labelled: LabelledTranscripts to convert. Callers should filter to
            ``included=True`` first; this function rejects any transcript
            with ``scratchpad=None`` because such transcripts are excluded
            from analysis by convention.

    Returns:
        One string per input, in the same order as the input list.

    Raises:
        ValueError: If any transcript has ``scratchpad=None``. Filter
            excluded transcripts with ``filter_included`` first.
    """
    texts: list[str] = []
    for item in labelled:
        if item.transcript.scratchpad is None:
            raise ValueError(
                "Transcript with no scratchpad encountered; "
                "call filter_included first to drop excluded transcripts."
            )
        texts.append(f"{item.transcript.scratchpad}\n\n{item.transcript.response}")
    return texts


def extract_activations(
    texts: list[str],
    config: ExtractionConfig,
) -> dict[int, np.ndarray]:
    """Extract mean-pooled residual stream activations from a list of texts.

    Runs one forward pass per batch and captures the residual stream
    output at each configured layer simultaneously. Mean-pools each captured
    tensor over the sequence dimension before returning.

    Apollo-format check is the caller's responsibility — see this module's
    docstring. NNSight API details vary across versions; the implementation
    below targets nnsight>=0.5,<1.0 (per pyproject.toml) and is validated
    only by manual smoke tests on RunPod.

    Args:
        texts: List of strings to extract activations from. For AF
            transcripts, build texts with :func:`get_af_extraction_texts`.
        config: Extraction configuration (model, layers, batch, quantisation).

    Returns:
        Dict mapping layer index → ndarray of shape ``(n_texts, hidden_dim)``.
        For Llama-3.3-70B-Instruct, ``hidden_dim`` is 8192.

    Raises:
        ValueError: If any layer in ``config.layers`` is out of range for
            the loaded model.
        RuntimeError: If a captured hidden state is not 3D (see
            :func:`_pool_hidden`).
    """
    from nnsight import LanguageModel

    quantisation_config = _build_quantisation_config(config.quantisation)
    model_kwargs: dict[str, object] = {"device_map": config.device}
    if quantisation_config is not None:
        model_kwargs["quantization_config"] = quantisation_config
    model = LanguageModel(config.model_name, **model_kwargs)

    n_layers = len(model.model.layers)
    invalid_layers = [layer for layer in config.layers if layer >= n_layers]
    if invalid_layers:
        raise ValueError(
            f"Layer indices {invalid_layers} out of range; model has {n_layers} layers"
        )

    per_layer_batches: dict[int, list[np.ndarray]] = {layer: [] for layer in config.layers}

    # Tokenisation with truncation happens explicitly so nnsight's trace
    # block doesn't have to know about max_length (the kwarg name varies
    # across nnsight versions). Passing pre-tokenised inputs makes the
    # truncation contract explicit and version-independent.
    tokenizer = model.tokenizer
    pad_token_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = pad_token_id

    for batch_start in range(0, len(texts), config.batch_size):
        batch = texts[batch_start : batch_start + config.batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_length,
        )
        # Pre-initialise captures so an exception swallowed by nnsight's
        # trace __exit__ surfaces as an empty-dict assertion rather than
        # an UnboundLocalError that hides the real failure.
        captured: dict[int, object] = {}
        with model.trace(encoded):
            for layer in config.layers:
                # Capture the layer's full output. nnsight versions differ on
                # whether `.output` is the hidden state tensor directly or the
                # raw module-return tuple `(hidden, ...)`. We handle both
                # below; never index with `[0]` here because newer nnsight
                # would treat that as a batch-dim index on the tensor.
                captured[layer] = model.model.layers[layer].output.save()
        if len(captured) != len(config.layers):
            raise RuntimeError(
                f"nnsight trace exited without populating captures for all "
                f"requested layers (got {sorted(captured.keys())}, expected "
                f"{config.layers}). Likely an API mismatch or model-structure "
                f"path issue in `model.model.layers[layer].output`."
            )
        for layer, save in captured.items():
            hidden = save.value if hasattr(save, "value") else save
            # Some nnsight / HF model combinations expose the layer output as
            # a tuple (hidden_state, attn_weights, ...); unwrap when needed.
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            assert isinstance(hidden, torch.Tensor), (
                f"Layer {layer}: expected torch.Tensor after nnsight capture, "
                f"got {type(hidden).__name__}"
            )
            per_layer_batches[layer].append(_pool_hidden(hidden, layer))

    return {layer: np.concatenate(per_layer_batches[layer], axis=0) for layer in config.layers}


def _build_quantisation_config(quantisation: str) -> object | None:
    """Build a ``BitsAndBytesConfig`` for the given quantisation string.

    Returns ``None`` for ``"none"`` so the model loads at its default
    precision. The bitsandbytes library is a Linux-only uv dependency (see
    ``pyproject.toml``) — quantised paths will fail on Windows / macOS
    where the wheel is unavailable, which is intended.
    """
    if quantisation == "none":
        return None
    from transformers import BitsAndBytesConfig

    if quantisation == "4bit":
        return BitsAndBytesConfig(  # type: ignore[no-untyped-call]
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    if quantisation == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)  # type: ignore[no-untyped-call]
    raise ValueError(f"Unknown quantisation: {quantisation}")
