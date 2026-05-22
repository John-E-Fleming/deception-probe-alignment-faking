"""Tests for activation extraction helpers.

The actual ``extract_activations`` function is not unit-tested here — it
requires a GPU and a real Llama-3.3 model. Its validation happens via a
manual smoke test on RunPod plus the required Apollo-format cross-check
documented in the module's docstring.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from alignment_faking_probes.data.activation_extraction import (
    ExtractionConfig,
    _pool_hidden,
    get_af_extraction_texts,
)
from alignment_faking_probes.data.labeling import LabelledTranscript
from alignment_faking_probes.data.transcript_generation import Transcript


def _make_labelled(
    scratchpad: str | None = "thinking about the problem",
    response: str = "the answer is 42",
    label: int = 1,
    included: bool = True,
) -> LabelledTranscript:
    return LabelledTranscript(
        transcript=Transcript(
            prompt="user prompt",
            scratchpad=scratchpad,
            response=response,
            metadata={},
        ),
        label=label,
        classifier_confidence=0.9,
        included=included,
    )


def test_pool_hidden_returns_correct_shape() -> None:
    hidden = torch.ones((2, 10, 8192))
    pooled = _pool_hidden(hidden, layer=40)
    assert pooled.shape == (2, 8192)


def test_pool_hidden_computes_mean_along_seq_dim() -> None:
    """Sequence-dim mean of a tensor with known structure should equal a
    sequence-wise average, not a batch-wise or feature-wise one."""
    # batch=1, seq=3, hidden=2. seq mean of [[1,2],[3,4],[5,6]] = [3, 4].
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    pooled = _pool_hidden(hidden, layer=0)
    np.testing.assert_allclose(pooled, np.array([[3.0, 4.0]]))


def test_pool_hidden_rejects_2d_tensor() -> None:
    hidden = torch.ones((10, 8192))
    with pytest.raises(RuntimeError, match="expected 3D tensor"):
        _pool_hidden(hidden, layer=40)


def test_pool_hidden_rejects_4d_tensor() -> None:
    hidden = torch.ones((2, 4, 10, 8192))
    with pytest.raises(RuntimeError, match="expected 3D tensor"):
        _pool_hidden(hidden, layer=40)


def test_get_af_extraction_texts_includes_scratchpad_and_response() -> None:
    labelled = [_make_labelled(scratchpad="my reasoning", response="final answer")]
    texts = get_af_extraction_texts(labelled)
    assert len(texts) == 1
    assert "my reasoning" in texts[0]
    assert "final answer" in texts[0]


def test_get_af_extraction_texts_preserves_order() -> None:
    labelled = [
        _make_labelled(scratchpad=f"scratch {i}", response=f"response {i}") for i in range(5)
    ]
    texts = get_af_extraction_texts(labelled)
    for i, text in enumerate(texts):
        assert f"scratch {i}" in text
        assert f"response {i}" in text


def test_get_af_extraction_texts_rejects_none_scratchpad() -> None:
    labelled = [_make_labelled(scratchpad=None)]
    with pytest.raises(ValueError, match="no scratchpad"):
        get_af_extraction_texts(labelled)


def test_extraction_config_construction() -> None:
    config = ExtractionConfig(
        model_name="meta-llama/Llama-3.3-70B-Instruct",
        layers=[20, 40, 60],
        batch_size=8,
        quantisation="4bit",
        device="cuda",
        max_length=2048,
    )
    assert config.layers == [20, 40, 60]
    assert config.quantisation == "4bit"


def test_extraction_config_rejects_unknown_quantisation() -> None:
    with pytest.raises(ValueError, match="quantisation"):
        ExtractionConfig(
            model_name="x",
            layers=[40],
            batch_size=1,
            quantisation="3bit",
            device="cuda",
            max_length=2048,
        )


def test_extraction_config_rejects_empty_layers() -> None:
    with pytest.raises(ValueError, match="layers must be non-empty"):
        ExtractionConfig(
            model_name="x",
            layers=[],
            batch_size=1,
            quantisation="none",
            device="cuda",
            max_length=2048,
        )


def test_extraction_config_rejects_negative_layers() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ExtractionConfig(
            model_name="x",
            layers=[40, -1, 60],
            batch_size=1,
            quantisation="none",
            device="cuda",
            max_length=2048,
        )


def test_extraction_config_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        ExtractionConfig(
            model_name="x",
            layers=[40],
            batch_size=0,
            quantisation="none",
            device="cuda",
            max_length=2048,
        )


def test_extraction_config_rejects_non_positive_max_length() -> None:
    with pytest.raises(ValueError, match="max_length"):
        ExtractionConfig(
            model_name="x",
            layers=[40],
            batch_size=1,
            quantisation="none",
            device="cuda",
            max_length=0,
        )
