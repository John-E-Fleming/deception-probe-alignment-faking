"""Tests for ProbeResult training, prediction, and serialisation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from alignment_faking_probes import APOLLO_ROLEPLAYING
from alignment_faking_probes.data.activation_store import ActivationDataset
from alignment_faking_probes.probes.training import (
    load_probe,
    predict_probe,
    save_probe,
    train_probe,
)


def _make_dataset(
    n_samples: int = 30,
    hidden_dim: int = 8,
    scenario: str = APOLLO_ROLEPLAYING,
    seed: int = 0,
    separable: bool = True,
) -> ActivationDataset:
    """Build a synthetic ActivationDataset.

    With ``separable=True`` the positive class is offset in activation space
    so a linear probe can learn it. With ``separable=False`` the labels are
    random with respect to activations — used to check that probes do not
    just memorise.
    """
    rng = np.random.default_rng(seed)
    labels = np.array([0, 1] * (n_samples // 2), dtype=np.int64)
    activations = rng.standard_normal((n_samples, hidden_dim)).astype(np.float32)
    if separable:
        activations[labels == 1] += 1.5
    return ActivationDataset(
        activations=activations,
        labels=labels,
        scenario=scenario,
        layer=40,
        model_name="meta-llama/Llama-3.3-70B-Instruct",
        n_positive=int((labels == 1).sum()),
        n_negative=int((labels == 0).sum()),
        metadata={"split": "train"},
    )


def test_train_probe_returns_probe_result_with_correct_metadata() -> None:
    dataset = _make_dataset(n_samples=30, hidden_dim=8)
    result = train_probe(dataset, seed=42)
    assert result.train_scenario == APOLLO_ROLEPLAYING
    assert result.layer == 40
    assert result.n_train == 30
    assert result.n_positive == 15
    assert isinstance(result.probe, LogisticRegression)
    assert isinstance(result.scaler, StandardScaler)
    assert 0.0 <= result.train_auroc <= 1.0
    assert result.metadata["model_name"] == "meta-llama/Llama-3.3-70B-Instruct"
    assert result.metadata["seed"] == "42"


def test_train_probe_achieves_high_auroc_on_separable_data() -> None:
    dataset = _make_dataset(n_samples=50, separable=True)
    result = train_probe(dataset, seed=42)
    # With a 1.5σ class offset and 50 samples, LOOCV AUROC should be high.
    # 0.85 is a generous lower bound; actual is usually 0.95+.
    assert result.train_auroc >= 0.85


def test_train_probe_raises_with_too_few_positives() -> None:
    rng = np.random.default_rng(0)
    activations = rng.standard_normal((10, 4)).astype(np.float32)
    labels = np.array([0] * 9 + [1], dtype=np.int64)
    dataset = ActivationDataset(
        activations=activations,
        labels=labels,
        scenario=APOLLO_ROLEPLAYING,
        layer=40,
        model_name="meta-llama/Llama-3.3-70B-Instruct",
        n_positive=1,
        n_negative=9,
        metadata={},
    )
    with pytest.raises(ValueError, match="at least 2 positive"):
        train_probe(dataset, seed=42)


def test_predict_probe_returns_probabilities() -> None:
    dataset = _make_dataset()
    result = train_probe(dataset, seed=42)
    scores = predict_probe(result, dataset.activations)
    assert scores.shape == (dataset.activations.shape[0],)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)


def test_predict_probe_raises_on_wrong_feature_dim() -> None:
    dataset = _make_dataset()
    result = train_probe(dataset, seed=42)
    wrong_shape = np.zeros((5, dataset.activations.shape[1] + 1), dtype=np.float32)
    with pytest.raises(ValueError, match="features"):
        predict_probe(result, wrong_shape)


def test_predict_probe_uses_training_scaler() -> None:
    """The bundled scaler must be used at inference time.

    If a fresh scaler were fit on the inference data each call, the same
    activations multiplied by 100 would normalise to the same z-scores and
    produce the same predictions. The fact that they don't confirms the
    training scaler is being applied verbatim — which is the whole point
    of bundling it with the probe.
    """
    dataset = _make_dataset()
    result = train_probe(dataset, seed=42)
    scores_original = predict_probe(result, dataset.activations)
    scores_scaled = predict_probe(result, dataset.activations * 100)
    assert not np.allclose(scores_original, scores_scaled, atol=0.01)


def test_save_probe_load_probe_round_trip(tmp_path: Path) -> None:
    dataset = _make_dataset()
    result = train_probe(dataset, seed=42)
    save_probe(result, tmp_path / "probe.pkl")
    loaded = load_probe(tmp_path / "probe.pkl")

    assert isinstance(loaded.probe, LogisticRegression)
    assert isinstance(loaded.scaler, StandardScaler)
    assert loaded.train_scenario == result.train_scenario
    assert loaded.layer == result.layer
    assert loaded.train_auroc == result.train_auroc
    assert loaded.n_train == result.n_train
    assert loaded.n_positive == result.n_positive

    # Predictions must be identical after round-trip — this is what
    # guarantees the scaler came back with its training fit intact.
    scores_original = predict_probe(result, dataset.activations)
    scores_loaded = predict_probe(loaded, dataset.activations)
    np.testing.assert_allclose(scores_original, scores_loaded)


def test_save_probe_raises_on_missing_parent_dir(tmp_path: Path) -> None:
    dataset = _make_dataset()
    result = train_probe(dataset, seed=42)
    with pytest.raises(FileNotFoundError, match="Parent directory"):
        save_probe(result, tmp_path / "missing_subdir" / "probe.pkl")


def test_load_probe_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=".pkl not found"):
        load_probe(tmp_path / "does_not_exist.pkl")


def test_load_probe_raises_on_wrong_object_type(tmp_path: Path) -> None:
    import pickle

    path = tmp_path / "not_a_probe.pkl"
    with path.open("wb") as f:
        pickle.dump({"not": "a probe"}, f)
    with pytest.raises(ValueError, match="not a ProbeResult"):
        load_probe(path)
