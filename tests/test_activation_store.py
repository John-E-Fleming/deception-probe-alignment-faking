"""Tests for ActivationDataset save/load round-trips and validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alignment_faking_probes import ALIGNMENT_FAKING, APOLLO_ROLEPLAYING
from alignment_faking_probes.data.activation_store import (
    ActivationDataset,
    load_activation_dataset,
    save_activation_dataset,
)


def _make_dataset(
    n_samples: int = 8,
    hidden_dim: int = 16,
    scenario: str = APOLLO_ROLEPLAYING,
    layer: int = 40,
    metadata: dict[str, str] | None = None,
) -> ActivationDataset:
    rng = np.random.default_rng(seed=0)
    activations = rng.standard_normal((n_samples, hidden_dim)).astype(np.float32)
    labels = np.array([1, 0] * (n_samples // 2), dtype=np.int64)
    return ActivationDataset(
        activations=activations,
        labels=labels,
        scenario=scenario,
        layer=layer,
        model_name="meta-llama/Llama-3.3-70B-Instruct",
        n_positive=int((labels == 1).sum()),
        n_negative=int((labels == 0).sum()),
        metadata=metadata if metadata is not None else {"split": "train"},
    )


def test_round_trip_preserves_arrays(tmp_path: Path) -> None:
    dataset = _make_dataset()
    save_activation_dataset(dataset, tmp_path / "test.npz")
    loaded = load_activation_dataset(tmp_path / "test.npz")
    np.testing.assert_array_equal(loaded.activations, dataset.activations)
    np.testing.assert_array_equal(loaded.labels, dataset.labels)
    assert loaded.scenario == dataset.scenario
    assert loaded.layer == dataset.layer
    assert loaded.model_name == dataset.model_name
    assert loaded.n_positive == dataset.n_positive
    assert loaded.n_negative == dataset.n_negative


def test_metadata_preserved_after_round_trip(tmp_path: Path) -> None:
    metadata = {
        "split": "train",
        "apollo_commit": "f8ec4010e74927394709dffa22b97bdf8cd5a62f",
        "extraction_date": "2026-05-22",
        "quantisation": "bf16",
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
            scenario="not_a_real_scenario",
            layer=40,
            model_name="meta-llama/Llama-3.3-70B-Instruct",
            n_positive=2,
            n_negative=2,
        )


def test_construction_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="mismatched lengths"):
        ActivationDataset(
            activations=np.zeros((4, 8)),
            labels=np.array([0, 1, 0]),
            scenario=APOLLO_ROLEPLAYING,
            layer=40,
            model_name="meta-llama/Llama-3.3-70B-Instruct",
            n_positive=1,
            n_negative=2,
        )


def test_construction_rejects_inconsistent_n_positive() -> None:
    with pytest.raises(ValueError, match="n_positive"):
        ActivationDataset(
            activations=np.zeros((4, 8)),
            labels=np.array([0, 1, 0, 1]),
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
            scenario=APOLLO_ROLEPLAYING,
            layer=40,
            model_name="meta-llama/Llama-3.3-70B-Instruct",
            n_positive=2,
            n_negative=10,
        )


def test_load_raises_filenotfound_on_missing_npz(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=".npz not found"):
        load_activation_dataset(tmp_path / "does_not_exist.npz")


def test_load_raises_filenotfound_on_missing_sidecar(tmp_path: Path) -> None:
    np.savez_compressed(
        tmp_path / "orphan.npz",
        activations=np.zeros((4, 8)),
        labels=np.array([0, 1, 0, 1]),
    )
    with pytest.raises(FileNotFoundError, match="sidecar not found"):
        load_activation_dataset(tmp_path / "orphan.npz")


def test_save_raises_filenotfound_on_missing_parent_dir(tmp_path: Path) -> None:
    dataset = _make_dataset()
    bad_path = tmp_path / "nonexistent_subdir" / "test.npz"
    with pytest.raises(FileNotFoundError, match="Parent directory"):
        save_activation_dataset(dataset, bad_path)
