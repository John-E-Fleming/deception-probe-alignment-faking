"""Tests for ActivationDataset save/load round-trips and validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from alignment_faking_probes import ALIGNMENT_FAKING, APOLLO_ROLEPLAYING
from alignment_faking_probes.data.activation_store import (
    SIDECAR_FORMAT_VERSION,
    ActivationDataset,
    load_activation_dataset,
    save_activation_dataset,
)


def _make_dataset(
    n_samples: int = 8,
    hidden_dim: int = 16,
    tokens_per_sample: int = 1,
    scenario: str = APOLLO_ROLEPLAYING,
    layer: int = 40,
    metadata: dict[str, str] | None = None,
) -> ActivationDataset:
    """Build a synthetic per-token ActivationDataset.

    Defaults to ``tokens_per_sample=1`` so older tests that pre-date the
    per-token refactor remain semantically identical (one row = one sample).
    Per-sample labels alternate ``[1, 0, 1, 0, ...]`` and are broadcast
    across the sample's tokens.
    """
    rng = np.random.default_rng(seed=0)
    n_tokens = n_samples * tokens_per_sample
    activations = rng.standard_normal((n_tokens, hidden_dim)).astype(np.float32)
    sample_labels = np.array([1, 0] * (n_samples // 2), dtype=np.int64)
    sample_id = np.repeat(np.arange(n_samples, dtype=np.int64), tokens_per_sample)
    labels = np.repeat(sample_labels, tokens_per_sample)
    return ActivationDataset(
        activations=activations,
        labels=labels,
        sample_id=sample_id,
        scenario=scenario,
        layer=layer,
        model_name="meta-llama/Llama-3.3-70B-Instruct",
        n_positive=int((sample_labels == 1).sum()),
        n_negative=int((sample_labels == 0).sum()),
        metadata=metadata if metadata is not None else {"split": "train"},
    )


def test_round_trip_preserves_arrays(tmp_path: Path) -> None:
    dataset = _make_dataset()
    save_activation_dataset(dataset, tmp_path / "test.npz")
    loaded = load_activation_dataset(tmp_path / "test.npz")
    np.testing.assert_array_equal(loaded.activations, dataset.activations)
    np.testing.assert_array_equal(loaded.labels, dataset.labels)
    np.testing.assert_array_equal(loaded.sample_id, dataset.sample_id)
    assert loaded.scenario == dataset.scenario
    assert loaded.layer == dataset.layer
    assert loaded.model_name == dataset.model_name
    assert loaded.n_positive == dataset.n_positive
    assert loaded.n_negative == dataset.n_negative


def test_round_trip_preserves_arrays_with_multiple_tokens_per_sample(tmp_path: Path) -> None:
    """The per-token refactor matters; exercise the >1 tokens-per-sample path."""
    dataset = _make_dataset(n_samples=6, tokens_per_sample=4)
    save_activation_dataset(dataset, tmp_path / "test.npz")
    loaded = load_activation_dataset(tmp_path / "test.npz")
    assert loaded.activations.shape == (24, 16)
    np.testing.assert_array_equal(loaded.sample_id, dataset.sample_id)
    # n_positive and n_negative are per-sample, not per-token.
    assert loaded.n_positive == 3
    assert loaded.n_negative == 3


def test_metadata_preserved_after_round_trip(tmp_path: Path) -> None:
    metadata = {
        "split": "train",
        "apollo_commit": "f8ec4010e74927394709dffa22b97bdf8cd5a62f",
        "extraction_date": "2026-05-22",
        "detection_mask_kind": "both",
    }
    dataset = _make_dataset(metadata=metadata)
    save_activation_dataset(dataset, tmp_path / "test.npz")
    loaded = load_activation_dataset(tmp_path / "test.npz")
    assert loaded.metadata == metadata


def test_alignment_faking_scenario_accepted() -> None:
    dataset = _make_dataset(scenario=ALIGNMENT_FAKING)
    assert dataset.scenario == ALIGNMENT_FAKING


def test_construction_rejects_unknown_scenario() -> None:
    with pytest.raises(ValueError, match="Unrecognised scenario"):
        ActivationDataset(
            activations=np.zeros((4, 8)),
            labels=np.array([0, 1, 0, 1]),
            sample_id=np.array([0, 1, 2, 3]),
            scenario="not_a_real_scenario",
            layer=40,
            model_name="meta-llama/Llama-3.3-70B-Instruct",
            n_positive=2,
            n_negative=2,
        )


def test_construction_rejects_mismatched_label_length() -> None:
    with pytest.raises(ValueError, match="activations and labels have mismatched lengths"):
        ActivationDataset(
            activations=np.zeros((4, 8)),
            labels=np.array([0, 1, 0]),
            sample_id=np.array([0, 1, 2, 3]),
            scenario=APOLLO_ROLEPLAYING,
            layer=40,
            model_name="meta-llama/Llama-3.3-70B-Instruct",
            n_positive=1,
            n_negative=2,
        )


def test_construction_rejects_mismatched_sample_id_length() -> None:
    with pytest.raises(ValueError, match="activations and sample_id have mismatched lengths"):
        ActivationDataset(
            activations=np.zeros((4, 8)),
            labels=np.array([0, 1, 0, 1]),
            sample_id=np.array([0, 1, 2]),
            scenario=APOLLO_ROLEPLAYING,
            layer=40,
            model_name="meta-llama/Llama-3.3-70B-Instruct",
            n_positive=2,
            n_negative=2,
        )


def test_construction_rejects_non_integer_sample_id() -> None:
    with pytest.raises(ValueError, match="sample_id must be an integer array"):
        ActivationDataset(
            activations=np.zeros((4, 8)),
            labels=np.array([0, 1, 0, 1]),
            sample_id=np.array([0.0, 1.0, 2.0, 3.0]),
            scenario=APOLLO_ROLEPLAYING,
            layer=40,
            model_name="meta-llama/Llama-3.3-70B-Instruct",
            n_positive=2,
            n_negative=2,
        )


def test_construction_rejects_noncontiguous_sample_ids() -> None:
    with pytest.raises(ValueError, match="contiguous integers from 0"):
        ActivationDataset(
            activations=np.zeros((4, 8)),
            labels=np.array([0, 1, 0, 1]),
            sample_id=np.array([0, 1, 3, 4]),
            scenario=APOLLO_ROLEPLAYING,
            layer=40,
            model_name="meta-llama/Llama-3.3-70B-Instruct",
            n_positive=2,
            n_negative=2,
        )


def test_construction_rejects_mixed_labels_within_sample() -> None:
    """Tokens within one sample must share a single label — broadcast invariant."""
    with pytest.raises(ValueError, match="Sample 0 has mixed labels"):
        ActivationDataset(
            activations=np.zeros((4, 8)),
            labels=np.array([0, 1, 0, 1]),  # sample 0 has [0, 1]
            sample_id=np.array([0, 0, 1, 1]),
            scenario=APOLLO_ROLEPLAYING,
            layer=40,
            model_name="meta-llama/Llama-3.3-70B-Instruct",
            n_positive=1,
            n_negative=1,
        )


def test_construction_rejects_inconsistent_n_positive() -> None:
    with pytest.raises(ValueError, match="n_positive"):
        ActivationDataset(
            activations=np.zeros((4, 8)),
            labels=np.array([0, 1, 0, 1]),
            sample_id=np.array([0, 1, 2, 3]),
            scenario=APOLLO_ROLEPLAYING,
            layer=40,
            model_name="meta-llama/Llama-3.3-70B-Instruct",
            n_positive=10,
            n_negative=2,
        )


def test_construction_rejects_inconsistent_n_negative() -> None:
    with pytest.raises(ValueError, match="n_negative"):
        ActivationDataset(
            activations=np.zeros((4, 8)),
            labels=np.array([0, 1, 0, 1]),
            sample_id=np.array([0, 1, 2, 3]),
            scenario=APOLLO_ROLEPLAYING,
            layer=40,
            model_name="meta-llama/Llama-3.3-70B-Instruct",
            n_positive=2,
            n_negative=10,
        )


def test_n_positive_counts_samples_not_tokens() -> None:
    """With multiple tokens per sample, n_positive is the per-sample count."""
    dataset = _make_dataset(n_samples=6, tokens_per_sample=5)
    # 6 samples, alternating labels [1, 0, 1, 0, 1, 0] → 3 positive, 3 negative
    # samples; 30 tokens total. n_positive is 3, not 15.
    assert dataset.n_positive == 3
    assert dataset.n_negative == 3
    assert dataset.activations.shape[0] == 30


def test_unicode_metadata_round_trips(tmp_path: Path) -> None:
    """Non-ASCII metadata must survive save/load on any platform.

    On Windows, ``Path.write_text`` defaults to the system encoding (often
    cp1252) rather than UTF-8. Explicit ``encoding="utf-8"`` in save/load
    ensures sidecars written on Windows can be read on RunPod (Linux) and
    vice versa.
    """
    metadata = {"note": "café résultat 中文"}
    dataset = _make_dataset(metadata=metadata)
    save_activation_dataset(dataset, tmp_path / "unicode.npz")
    loaded = load_activation_dataset(tmp_path / "unicode.npz")
    assert loaded.metadata == metadata


def test_load_raises_filenotfound_on_missing_npz(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=".npz not found"):
        load_activation_dataset(tmp_path / "does_not_exist.npz")


def test_load_raises_filenotfound_on_missing_sidecar(tmp_path: Path) -> None:
    np.savez_compressed(
        tmp_path / "orphan.npz",
        activations=np.zeros((4, 8)),
        labels=np.array([0, 1, 0, 1]),
        sample_id=np.array([0, 1, 2, 3]),
    )
    with pytest.raises(FileNotFoundError, match="sidecar not found"):
        load_activation_dataset(tmp_path / "orphan.npz")


def test_load_rejects_v1_format(tmp_path: Path) -> None:
    """Old per-sample files (no format_version or version != 2) must be rejected.

    There is no in-place upgrade path — re-extracting is the only valid
    response to encountering a v1 file. Silently accepting it would let
    pre-refactor activations slip into the per-token pipeline.
    """
    # Mimic a v1 sidecar (no format_version key, no sample_id in npz)
    np.savez_compressed(
        tmp_path / "old.npz",
        activations=np.zeros((4, 8)),
        labels=np.array([0, 1, 0, 1]),
    )
    sidecar = {
        "scenario": APOLLO_ROLEPLAYING,
        "layer": 40,
        "model_name": "meta-llama/Llama-3.3-70B-Instruct",
        "n_positive": 2,
        "n_negative": 2,
        "metadata": {},
    }
    (tmp_path / "old.json").write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(ValueError, match="format_version"):
        load_activation_dataset(tmp_path / "old.npz")


def test_load_rejects_missing_sample_id_in_npz(tmp_path: Path) -> None:
    """A current-format sidecar but a .npz missing sample_id should error clearly."""
    np.savez_compressed(
        tmp_path / "broken.npz",
        activations=np.zeros((4, 8)),
        labels=np.array([0, 1, 0, 1]),
    )
    sidecar = {
        "format_version": SIDECAR_FORMAT_VERSION,
        "scenario": APOLLO_ROLEPLAYING,
        "layer": 40,
        "model_name": "meta-llama/Llama-3.3-70B-Instruct",
        "n_positive": 2,
        "n_negative": 2,
        "metadata": {},
    }
    (tmp_path / "broken.json").write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(ValueError, match="missing 'sample_id'"):
        load_activation_dataset(tmp_path / "broken.npz")


def test_save_raises_filenotfound_on_missing_parent_dir(tmp_path: Path) -> None:
    dataset = _make_dataset()
    bad_path = tmp_path / "nonexistent_subdir" / "test.npz"
    with pytest.raises(FileNotFoundError, match="Parent directory"):
        save_activation_dataset(dataset, bad_path)
