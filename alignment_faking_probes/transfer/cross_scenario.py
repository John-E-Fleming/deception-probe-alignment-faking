"""Cross-scenario probe transfer experiment.

A ``TransferCell`` is a single (train_scenario, eval_scenario) result: a probe
trained on one scenario's activations, applied to another's, with metrics
recorded. ``run_transfer_matrix`` produces every cross-scenario ordered pair
of the four recognised scenarios — N*(N-1) = 12 cells.

The 2×2 framing in the README is a conceptual aggregation built downstream
(in ``scripts/05_run_transfer_experiment.py`` and the plot helper here);
this module produces the raw 4×4 minus the within-distribution diagonal.
Within-distribution AUROC comes from ``ProbeResult.train_auroc`` in
``probes/training.py`` and is not computed here.

The critical invariant ``train_dataset.scenario != eval_dataset.scenario``
is enforced in ``run_transfer_cell`` — never the caller's responsibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from alignment_faking_probes import RECOGNISED_SCENARIOS
from alignment_faking_probes.data.activation_store import ActivationDataset
from alignment_faking_probes.probes.evaluation import ProbeMetrics, compute_metrics
from alignment_faking_probes.probes.training import (
    AggregationMethod,
    ProbeResult,
    aggregate_token_scores,
    predict_probe,
    train_probe,
)

#: Default per-token → per-sample aggregation used by the transfer matrix.
#: Matches Apollo's ``prompt_scorer.py`` so transfer-matrix AUROC values are
#: comparable to Apollo's published numbers. Recorded in
#: ``TransferMatrix.metadata["aggregation"]`` for the experiment log.
_DEFAULT_AGGREGATION: AggregationMethod = "mean"

_VALID_METRICS = ("auroc", "tpr_at_1pct_fpr", "tpr_at_5pct_fpr")


@dataclass
class TransferCell:
    """A single (train, eval) cross-scenario probe transfer result.

    ``probe_result`` is set to ``None`` after ``load_transfer_matrix`` —
    probes are not persisted in the matrix JSON, only their metadata. Live
    cells produced by ``run_transfer_cell`` always have it populated.

    Attributes:
        train_scenario: Scenario the probe was trained on.
        eval_scenario: Scenario the probe was evaluated on.
        layer: Residual stream layer index (must match between train and eval).
        metrics: AUROC, TPR@1%FPR, TPR@5%FPR, and class counts on the eval set.
        probe_result: The trained probe + scaler, or ``None`` after JSON load.
    """

    train_scenario: str
    eval_scenario: str
    layer: int
    metrics: ProbeMetrics
    probe_result: ProbeResult | None = None


@dataclass
class TransferMatrix:
    """All cross-scenario probe transfer results, plus run metadata.

    Attributes:
        cells: One TransferCell per cross-scenario ordered pair.
        apollo_commit: Pinned Apollo repo commit hash used at extraction time.
        model_name: Hugging Face model name shared across all cells.
        seed: Random seed used for all probe training.
        completed_at: ISO 8601 UTC timestamp of when the matrix finished.
    """

    cells: list[TransferCell]
    apollo_commit: str
    model_name: str
    seed: int
    completed_at: str
    metadata: dict[str, str] = field(default_factory=dict)

    def get_cell(
        self,
        train_scenario: str,
        eval_scenario: str,
        layer: int,
    ) -> TransferCell | None:
        """Look up a specific cell by (train, eval, layer)."""
        for cell in self.cells:
            if (
                cell.train_scenario == train_scenario
                and cell.eval_scenario == eval_scenario
                and cell.layer == layer
            ):
                return cell
        return None

    def to_auroc_table(self) -> dict[str, dict[str, float]]:
        """Return AUROC as a nested dict: train_scenario → eval_scenario → AUROC.

        Suitable for JSON serialisation and for feeding into
        ``plot_transfer_matrix_heatmap``.
        """
        table: dict[str, dict[str, float]] = {}
        for cell in self.cells:
            table.setdefault(cell.train_scenario, {})[cell.eval_scenario] = cell.metrics.auroc
        return table


def run_transfer_cell(
    train_dataset: ActivationDataset,
    eval_dataset: ActivationDataset,
    seed: int,
    aggregation: AggregationMethod = _DEFAULT_AGGREGATION,
) -> TransferCell:
    """Train a probe on ``train_dataset``, evaluate per-sample on ``eval_dataset``.

    The probe is trained at the token level (one row per token). At
    evaluation time, per-token scores are aggregated to per-sample using
    ``aggregation`` (mean by default — matches Apollo). All transfer-matrix
    metrics (AUROC, TPR@FPR) are computed at the sample level.

    Args:
        train_dataset: Per-token activations + labels for the training scenario.
        eval_dataset: Per-token activations + labels for a DIFFERENT scenario.
        seed: Random seed for probe training.
        aggregation: Per-token → per-sample aggregation method. Defaults to
            ``"mean"`` (matches Apollo's prompt_scorer); ``"max"`` and
            ``"last"`` are available for ablations.

    Returns:
        TransferCell with trained probe and per-sample evaluation metrics.

    Raises:
        ValueError: If ``train_dataset.scenario == eval_dataset.scenario``.
            (Within-distribution evaluation lives in ``training.py``.)
        ValueError: If train and eval layers differ. Cross-layer transfer is
            not in scope for this project.
    """
    if train_dataset.scenario == eval_dataset.scenario:
        raise ValueError(
            f"run_transfer_cell expects different scenarios; "
            f"got both as {train_dataset.scenario!r}. "
            "Within-distribution evaluation lives in training.py "
            "(ProbeResult.train_auroc)."
        )
    if train_dataset.layer != eval_dataset.layer:
        raise ValueError(
            f"Transfer across layers is out of scope; "
            f"train layer={train_dataset.layer}, eval layer={eval_dataset.layer}"
        )

    probe_result = train_probe(train_dataset, seed=seed)
    token_scores = predict_probe(probe_result, eval_dataset.activations)
    sample_scores, sample_ids = aggregate_token_scores(
        token_scores, eval_dataset.sample_id, method=aggregation
    )
    # Per-sample labels: take the first token's label for each sample. Per the
    # ActivationDataset invariant, every token within a sample shares one label,
    # so any token's label works. ``np.unique`` returns indices in sample_id
    # order, which matches the order aggregate_token_scores uses.
    _, first_idx = np.unique(eval_dataset.sample_id, return_index=True)
    sample_labels = eval_dataset.labels[first_idx]
    # Sanity: aggregate_token_scores returns scores indexed 0..n-1, which is
    # the same order as the sorted unique sample_ids. Assertion makes this
    # contract explicit at runtime.
    assert np.array_equal(sample_ids, np.arange(sample_ids.size))

    metrics = compute_metrics(
        labels=sample_labels,
        scores=sample_scores,
        train_scenario=train_dataset.scenario,
        eval_scenario=eval_dataset.scenario,
    )

    return TransferCell(
        train_scenario=train_dataset.scenario,
        eval_scenario=eval_dataset.scenario,
        layer=train_dataset.layer,
        metrics=metrics,
        probe_result=probe_result,
    )


def run_transfer_matrix(
    datasets: dict[str, ActivationDataset],
    seed: int,
    apollo_commit: str,
    aggregation: AggregationMethod = _DEFAULT_AGGREGATION,
) -> TransferMatrix:
    """Run every cross-scenario transfer cell.

    For each ordered pair of distinct scenarios, trains a probe on the
    training scenario and evaluates on the eval scenario. With the four
    recognised scenarios, this produces 4*3 = 12 cells. Within-distribution
    AUROC is NOT computed here — read it from ``train_probe`` output instead.

    Args:
        datasets: Dict mapping scenario name → ActivationDataset. Must
            contain exactly the four recognised scenario names. All datasets
            must be at the same residual stream layer and from the same
            model.
        seed: Random seed for all probe training runs.
        apollo_commit: Apollo repo commit hash, logged in the matrix metadata.
        aggregation: Per-token → per-sample aggregation, recorded in
            ``TransferMatrix.metadata["aggregation"]``. Default ``"mean"``.

    Returns:
        TransferMatrix with all 12 cross-scenario cells populated.

    Raises:
        ValueError: If ``datasets`` is missing any of the four recognised
            scenario names.
        ValueError: If the datasets are at different layers.
        ValueError: If the datasets come from different models.
    """
    missing = set(RECOGNISED_SCENARIOS) - set(datasets.keys())
    if missing:
        raise ValueError(
            f"Missing scenarios: {sorted(missing)}; expected all of {RECOGNISED_SCENARIOS}"
        )

    layers = {scenario: datasets[scenario].layer for scenario in RECOGNISED_SCENARIOS}
    unique_layers = set(layers.values())
    if len(unique_layers) != 1:
        raise ValueError(
            f"All datasets must be at the same layer for cross-scenario transfer; "
            f"got layers per scenario: {layers}"
        )

    models = {scenario: datasets[scenario].model_name for scenario in RECOGNISED_SCENARIOS}
    unique_models = set(models.values())
    if len(unique_models) != 1:
        raise ValueError(
            f"All datasets must come from the same model; got models per scenario: {models}"
        )

    cells: list[TransferCell] = []
    for train_scenario in RECOGNISED_SCENARIOS:
        for eval_scenario in RECOGNISED_SCENARIOS:
            if train_scenario == eval_scenario:
                continue
            cells.append(
                run_transfer_cell(
                    train_dataset=datasets[train_scenario],
                    eval_dataset=datasets[eval_scenario],
                    seed=seed,
                    aggregation=aggregation,
                )
            )

    return TransferMatrix(
        cells=cells,
        apollo_commit=apollo_commit,
        model_name=unique_models.pop(),
        seed=seed,
        completed_at=datetime.now(timezone.utc).isoformat(),
        metadata={"aggregation": aggregation},
    )


def save_transfer_matrix(matrix: TransferMatrix, path: Path) -> None:
    """Save a TransferMatrix to JSON.

    Trained probes are NOT included in the JSON — only the metrics and the
    lightweight probe metadata (LOOCV AUROC, n_train, n_positive). Save the
    probes separately via ``save_probe`` if you need to re-evaluate later.

    Args:
        matrix: The TransferMatrix to save.
        path: Output JSON path. Parent directory must already exist.

    Raises:
        FileNotFoundError: If ``path.parent`` does not exist.
    """
    path = Path(path)
    if not path.parent.exists():
        raise FileNotFoundError(f"Parent directory does not exist: {path.parent}")

    payload = {
        "apollo_commit": matrix.apollo_commit,
        "model_name": matrix.model_name,
        "seed": matrix.seed,
        "completed_at": matrix.completed_at,
        "metadata": matrix.metadata,
        "cells": [
            {
                "train_scenario": cell.train_scenario,
                "eval_scenario": cell.eval_scenario,
                "layer": cell.layer,
                "metrics": {
                    "auroc": cell.metrics.auroc,
                    "tpr_at_1pct_fpr": cell.metrics.tpr_at_1pct_fpr,
                    "tpr_at_5pct_fpr": cell.metrics.tpr_at_5pct_fpr,
                    "n_positive": cell.metrics.n_positive,
                    "n_negative": cell.metrics.n_negative,
                },
                "probe_metadata": (
                    {
                        "train_auroc": cell.probe_result.train_auroc,
                        "n_train": cell.probe_result.n_train,
                        "n_positive": cell.probe_result.n_positive,
                    }
                    if cell.probe_result is not None
                    else None
                ),
            }
            for cell in matrix.cells
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_transfer_matrix(path: Path) -> TransferMatrix:
    """Load a TransferMatrix previously written by ``save_transfer_matrix``.

    Loaded cells have ``probe_result=None`` — probes are not persisted in
    the matrix JSON. The probe_metadata in the JSON is read as informational
    only.

    Args:
        path: Path to the JSON file.

    Returns:
        TransferMatrix with cells restored; probe_result on each cell is ``None``.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Transfer matrix JSON not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))

    cells = [
        TransferCell(
            train_scenario=raw["train_scenario"],
            eval_scenario=raw["eval_scenario"],
            layer=raw["layer"],
            metrics=ProbeMetrics(
                auroc=raw["metrics"]["auroc"],
                tpr_at_1pct_fpr=raw["metrics"]["tpr_at_1pct_fpr"],
                tpr_at_5pct_fpr=raw["metrics"]["tpr_at_5pct_fpr"],
                n_positive=raw["metrics"]["n_positive"],
                n_negative=raw["metrics"]["n_negative"],
                train_scenario=raw["train_scenario"],
                eval_scenario=raw["eval_scenario"],
            ),
            probe_result=None,
        )
        for raw in payload["cells"]
    ]

    return TransferMatrix(
        cells=cells,
        apollo_commit=payload["apollo_commit"],
        model_name=payload["model_name"],
        seed=payload["seed"],
        completed_at=payload["completed_at"],
        metadata=payload.get("metadata", {}),
    )


def plot_transfer_matrix_heatmap(
    cells: list[TransferCell],
    metric: str = "auroc",
    ax: Axes | None = None,
) -> Figure:
    """Plot the transfer matrix as a colour heatmap.

    Renders an N×N grid of the chosen metric, where N is the number of
    distinct scenarios across the cells. Diagonal cells are NaN (gray)
    because cross-scenario transfer never includes the within-distribution
    case — fill them in from ``ProbeResult.train_auroc`` if you want a
    complete 4×4 display.

    Args:
        cells: TransferCells to plot. May be a full matrix or a subset.
        metric: Which ProbeMetrics field to display. One of
            ``"auroc"``, ``"tpr_at_1pct_fpr"``, ``"tpr_at_5pct_fpr"``.
        ax: Optional existing Axes. A new Figure is created when None.

    Returns:
        The matplotlib Figure containing the heatmap.

    Raises:
        ValueError: If ``metric`` is not one of the recognised metric names.
    """
    if metric not in _VALID_METRICS:
        raise ValueError(f"metric must be one of {_VALID_METRICS}, got {metric!r}")

    train_scenarios = sorted({cell.train_scenario for cell in cells})
    eval_scenarios = sorted({cell.eval_scenario for cell in cells})

    matrix = np.full((len(train_scenarios), len(eval_scenarios)), np.nan)
    for cell in cells:
        i = train_scenarios.index(cell.train_scenario)
        j = eval_scenarios.index(cell.eval_scenario)
        matrix[i, j] = getattr(cell.metrics, metric)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        # ax.figure can be a SubFigure in matplotlib's type stubs; this
        # function only supports top-level Figures. Narrow with an assert so
        # the failure mode is a clear AssertionError rather than a confusing
        # downstream error.
        figure_attr = ax.figure
        assert isinstance(figure_attr, Figure), (
            f"Expected ax.figure to be a Figure, got {type(figure_attr).__name__}"
        )
        fig = figure_attr

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        xticklabels=eval_scenarios,
        yticklabels=train_scenarios,
        ax=ax,
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": metric},
    )
    ax.set_xlabel("Eval scenario")
    ax.set_ylabel("Train scenario")
    return fig
