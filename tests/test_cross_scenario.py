"""Tests for cross-scenario probe transfer experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from alignment_faking_probes import (
    ALIGNMENT_FAKING,
    APOLLO_INSIDER_TRADING,
    APOLLO_ROLEPLAYING,
    APOLLO_SANDBAGGING,
    RECOGNISED_SCENARIOS,
)
from alignment_faking_probes.data.activation_store import ActivationDataset
from alignment_faking_probes.transfer.cross_scenario import (
    TransferCell,
    TransferMatrix,
    load_transfer_matrix,
    plot_transfer_matrix_heatmap,
    run_transfer_cell,
    run_transfer_matrix,
    save_transfer_matrix,
)


def _make_dataset(
    scenario: str,
    n_samples: int = 20,
    hidden_dim: int = 8,
    layer: int = 40,
    seed: int = 0,
    separable: bool = True,
) -> ActivationDataset:
    rng = np.random.default_rng(seed)
    labels = np.array([0, 1] * (n_samples // 2), dtype=np.int64)
    activations = rng.standard_normal((n_samples, hidden_dim)).astype(np.float32)
    if separable:
        activations[labels == 1] += 1.5
    sample_id = np.arange(n_samples, dtype=np.int64)
    return ActivationDataset(
        activations=activations,
        labels=labels,
        sample_id=sample_id,
        scenario=scenario,
        layer=layer,
        model_name="meta-llama/Llama-3.3-70B-Instruct",
        n_positive=int((labels == 1).sum()),
        n_negative=int((labels == 0).sum()),
        metadata={"split": "train"},
    )


def _make_all_four_datasets(layer: int = 40) -> dict[str, ActivationDataset]:
    return {
        scenario: _make_dataset(scenario=scenario, layer=layer, seed=i)
        for i, scenario in enumerate(RECOGNISED_SCENARIOS)
    }


def test_run_transfer_cell_returns_valid_result() -> None:
    train_dataset = _make_dataset(APOLLO_ROLEPLAYING, seed=0)
    eval_dataset = _make_dataset(ALIGNMENT_FAKING, seed=1)
    cell = run_transfer_cell(train_dataset, eval_dataset, seed=42)
    assert cell.train_scenario == APOLLO_ROLEPLAYING
    assert cell.eval_scenario == ALIGNMENT_FAKING
    assert cell.layer == 40
    assert cell.probe_result is not None
    assert 0.0 <= cell.metrics.auroc <= 1.0


def test_run_transfer_cell_rejects_same_scenario() -> None:
    train = _make_dataset(APOLLO_ROLEPLAYING, seed=0)
    evalsame = _make_dataset(APOLLO_ROLEPLAYING, seed=1)
    with pytest.raises(ValueError, match="different scenarios"):
        run_transfer_cell(train, evalsame, seed=42)


def test_run_transfer_cell_rejects_different_layers() -> None:
    train = _make_dataset(APOLLO_ROLEPLAYING, layer=40, seed=0)
    eval_other_layer = _make_dataset(ALIGNMENT_FAKING, layer=60, seed=1)
    with pytest.raises(ValueError, match="layers"):
        run_transfer_cell(train, eval_other_layer, seed=42)


def test_run_transfer_cell_train_set_is_not_contaminated() -> None:
    """The trained probe's n_train should equal the training set size, not the
    union of train + eval. Guards against an accidental concat bug."""
    train = _make_dataset(APOLLO_ROLEPLAYING, n_samples=20, seed=0)
    evalds = _make_dataset(ALIGNMENT_FAKING, n_samples=30, seed=1)
    cell = run_transfer_cell(train, evalds, seed=42)
    assert cell.probe_result is not None
    assert cell.probe_result.n_train == 20


def _make_multi_token_eval_dataset(
    scenario: str,
    n_samples: int,
    tokens_per_sample: int,
    seed: int,
) -> ActivationDataset:
    rng = np.random.default_rng(seed)
    sample_labels = np.array([0, 1] * (n_samples // 2), dtype=np.int64)
    sample_centroids = rng.standard_normal((n_samples, 8)).astype(np.float32)
    sample_centroids[sample_labels == 1] += 1.5
    activations = np.repeat(sample_centroids, tokens_per_sample, axis=0)
    activations = activations + rng.standard_normal(activations.shape).astype(np.float32) * 0.05
    sample_id = np.repeat(np.arange(n_samples, dtype=np.int64), tokens_per_sample)
    labels = np.repeat(sample_labels, tokens_per_sample)
    return ActivationDataset(
        activations=activations,
        labels=labels,
        sample_id=sample_id,
        scenario=scenario,
        layer=40,
        model_name="meta-llama/Llama-3.3-70B-Instruct",
        n_positive=int((sample_labels == 1).sum()),
        n_negative=int((sample_labels == 0).sum()),
        metadata={"split": "test"},
    )


def test_run_transfer_cell_metrics_count_samples_not_tokens() -> None:
    """With multi-token eval samples, metrics.n_positive must reflect samples.

    A naive (un-aggregated) path would report n_positive equal to the number
    of positive **tokens** (n_samples * tokens_per_sample / 2). This test
    fails the moment ``run_transfer_cell`` regresses to passing per-token
    labels to ``compute_metrics``.
    """
    train = _make_dataset(APOLLO_ROLEPLAYING, n_samples=20, seed=0)
    eval_multi = _make_multi_token_eval_dataset(
        ALIGNMENT_FAKING, n_samples=20, tokens_per_sample=4, seed=1
    )
    cell = run_transfer_cell(train, eval_multi, seed=42)
    # 20 eval samples, 10 positive — NOT 80 tokens / 40 positive tokens.
    assert cell.metrics.n_positive == 10
    assert cell.metrics.n_negative == 10


def test_run_transfer_matrix_records_aggregation_metadata() -> None:
    datasets = _make_all_four_datasets()
    matrix = run_transfer_matrix(datasets, seed=42, apollo_commit="abc123")
    assert matrix.metadata["aggregation"] == "mean"

    matrix_max = run_transfer_matrix(datasets, seed=42, apollo_commit="abc123", aggregation="max")
    assert matrix_max.metadata["aggregation"] == "max"


def test_run_transfer_matrix_produces_all_cross_scenario_pairs() -> None:
    datasets = _make_all_four_datasets()
    matrix = run_transfer_matrix(datasets, seed=42, apollo_commit="abc123")
    # 4 scenarios, 4*3 = 12 cross-scenario ordered pairs (no diagonals).
    assert len(matrix.cells) == 12
    pairs = {(c.train_scenario, c.eval_scenario) for c in matrix.cells}
    assert (APOLLO_ROLEPLAYING, ALIGNMENT_FAKING) in pairs
    assert (ALIGNMENT_FAKING, APOLLO_ROLEPLAYING) in pairs
    assert (APOLLO_INSIDER_TRADING, APOLLO_SANDBAGGING) in pairs
    # No diagonal cells
    for cell in matrix.cells:
        assert cell.train_scenario != cell.eval_scenario


def test_run_transfer_matrix_records_metadata() -> None:
    datasets = _make_all_four_datasets()
    matrix = run_transfer_matrix(datasets, seed=42, apollo_commit="abc123")
    assert matrix.apollo_commit == "abc123"
    assert matrix.model_name == "meta-llama/Llama-3.3-70B-Instruct"
    assert matrix.seed == 42
    assert matrix.completed_at  # ISO 8601 string


def test_run_transfer_matrix_rejects_missing_scenarios() -> None:
    datasets = {
        APOLLO_ROLEPLAYING: _make_dataset(APOLLO_ROLEPLAYING, seed=0),
        ALIGNMENT_FAKING: _make_dataset(ALIGNMENT_FAKING, seed=1),
    }
    with pytest.raises(ValueError, match="Missing scenarios"):
        run_transfer_matrix(datasets, seed=42, apollo_commit="abc123")


def test_run_transfer_matrix_rejects_mismatched_layers() -> None:
    datasets = _make_all_four_datasets(layer=40)
    datasets[ALIGNMENT_FAKING] = _make_dataset(ALIGNMENT_FAKING, layer=60, seed=99)
    with pytest.raises(ValueError, match="same layer"):
        run_transfer_matrix(datasets, seed=42, apollo_commit="abc123")


def test_transfer_matrix_get_cell_returns_matching_cell() -> None:
    datasets = _make_all_four_datasets()
    matrix = run_transfer_matrix(datasets, seed=42, apollo_commit="abc123")
    cell = matrix.get_cell(APOLLO_ROLEPLAYING, ALIGNMENT_FAKING, layer=40)
    assert cell is not None
    assert cell.train_scenario == APOLLO_ROLEPLAYING
    assert cell.eval_scenario == ALIGNMENT_FAKING


def test_transfer_matrix_get_cell_returns_none_for_missing() -> None:
    datasets = _make_all_four_datasets()
    matrix = run_transfer_matrix(datasets, seed=42, apollo_commit="abc123")
    assert matrix.get_cell(APOLLO_ROLEPLAYING, APOLLO_ROLEPLAYING, layer=40) is None
    assert matrix.get_cell(APOLLO_ROLEPLAYING, ALIGNMENT_FAKING, layer=99) is None


def test_transfer_matrix_to_auroc_table_has_all_pairs() -> None:
    datasets = _make_all_four_datasets()
    matrix = run_transfer_matrix(datasets, seed=42, apollo_commit="abc123")
    table = matrix.to_auroc_table()
    assert set(table.keys()) == set(RECOGNISED_SCENARIOS)
    for train_scenario in RECOGNISED_SCENARIOS:
        eval_scenarios = set(table[train_scenario].keys())
        expected = set(RECOGNISED_SCENARIOS) - {train_scenario}
        assert eval_scenarios == expected


def test_save_load_transfer_matrix_round_trip(tmp_path: Path) -> None:
    datasets = _make_all_four_datasets()
    matrix = run_transfer_matrix(datasets, seed=42, apollo_commit="abc123")
    save_transfer_matrix(matrix, tmp_path / "matrix.json")
    loaded = load_transfer_matrix(tmp_path / "matrix.json")

    assert loaded.apollo_commit == matrix.apollo_commit
    assert loaded.model_name == matrix.model_name
    assert loaded.seed == matrix.seed
    assert loaded.completed_at == matrix.completed_at
    assert len(loaded.cells) == len(matrix.cells)
    # Metrics survive intact
    for original, restored in zip(
        sorted(matrix.cells, key=lambda c: (c.train_scenario, c.eval_scenario)),
        sorted(loaded.cells, key=lambda c: (c.train_scenario, c.eval_scenario)),
        strict=True,
    ):
        assert restored.metrics.auroc == original.metrics.auroc
        assert restored.metrics.tpr_at_1pct_fpr == original.metrics.tpr_at_1pct_fpr
        assert restored.metrics.tpr_at_5pct_fpr == original.metrics.tpr_at_5pct_fpr


def test_loaded_transfer_matrix_has_no_probes(tmp_path: Path) -> None:
    """Loaded matrices intentionally have probe_result=None — probes are not
    persisted in the JSON. Load them separately from .pkl if needed."""
    datasets = _make_all_four_datasets()
    matrix = run_transfer_matrix(datasets, seed=42, apollo_commit="abc123")
    save_transfer_matrix(matrix, tmp_path / "matrix.json")
    loaded = load_transfer_matrix(tmp_path / "matrix.json")
    assert all(cell.probe_result is None for cell in loaded.cells)


def test_unicode_metadata_round_trips(tmp_path: Path) -> None:
    """Unicode in matrix metadata must survive save/load (Windows ↔ RunPod portability)."""
    datasets = _make_all_four_datasets()
    matrix = run_transfer_matrix(datasets, seed=42, apollo_commit="abc123")
    matrix.metadata["note"] = "café résultat 中文"
    save_transfer_matrix(matrix, tmp_path / "matrix.json")
    loaded = load_transfer_matrix(tmp_path / "matrix.json")
    assert loaded.metadata["note"] == "café résultat 中文"


def test_save_transfer_matrix_raises_on_missing_parent_dir(tmp_path: Path) -> None:
    datasets = _make_all_four_datasets()
    matrix = run_transfer_matrix(datasets, seed=42, apollo_commit="abc123")
    with pytest.raises(FileNotFoundError, match="Parent directory"):
        save_transfer_matrix(matrix, tmp_path / "missing_subdir" / "matrix.json")


def test_load_transfer_matrix_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_transfer_matrix(tmp_path / "does_not_exist.json")


def test_plot_transfer_matrix_heatmap_smoke() -> None:
    datasets = _make_all_four_datasets()
    matrix = run_transfer_matrix(datasets, seed=42, apollo_commit="abc123")
    fig = plot_transfer_matrix_heatmap(matrix.cells, metric="auroc")
    try:
        assert fig is not None
    finally:
        plt.close(fig)


def test_plot_transfer_matrix_heatmap_rejects_invalid_metric() -> None:
    cell = TransferCell(
        train_scenario=APOLLO_ROLEPLAYING,
        eval_scenario=ALIGNMENT_FAKING,
        layer=40,
        metrics=_dummy_metrics(),
    )
    with pytest.raises(ValueError, match="metric"):
        plot_transfer_matrix_heatmap([cell], metric="not_a_metric")


def _dummy_metrics():
    from alignment_faking_probes.probes.evaluation import ProbeMetrics

    return ProbeMetrics(
        auroc=0.5,
        tpr_at_1pct_fpr=0.0,
        tpr_at_5pct_fpr=0.0,
        n_positive=1,
        n_negative=1,
        train_scenario=APOLLO_ROLEPLAYING,
        eval_scenario=ALIGNMENT_FAKING,
    )


def test_transfer_matrix_dataclass_constructible_directly() -> None:
    matrix = TransferMatrix(
        cells=[],
        apollo_commit="abc",
        model_name="meta-llama/Llama-3.3-70B-Instruct",
        seed=42,
        completed_at="2026-05-22T00:00:00+00:00",
    )
    assert matrix.cells == []
    assert matrix.apollo_commit == "abc"
