"""Tests for probe evaluation metrics and ROC plotting."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from alignment_faking_probes import ALIGNMENT_FAKING, APOLLO_ROLEPLAYING
from alignment_faking_probes.probes.evaluation import (
    compute_metrics,
    plot_roc_curve,
    tpr_at_fpr,
)


def test_compute_metrics_on_perfect_predictions() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    metrics = compute_metrics(labels, scores, APOLLO_ROLEPLAYING, ALIGNMENT_FAKING)
    assert metrics.auroc == 1.0
    assert metrics.tpr_at_1pct_fpr == 1.0
    assert metrics.tpr_at_5pct_fpr == 1.0
    assert metrics.n_positive == 3
    assert metrics.n_negative == 3
    assert metrics.train_scenario == APOLLO_ROLEPLAYING
    assert metrics.eval_scenario == ALIGNMENT_FAKING


def test_compute_metrics_on_inverted_predictions() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    metrics = compute_metrics(labels, scores, APOLLO_ROLEPLAYING, ALIGNMENT_FAKING)
    assert metrics.auroc == 0.0
    assert metrics.tpr_at_1pct_fpr == 0.0


def test_compute_metrics_on_random_scores_is_approximately_half() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=200)
    # Guarantee both classes are present.
    labels[0] = 0
    labels[1] = 1
    scores = rng.uniform(0, 1, size=200)
    metrics = compute_metrics(labels, scores, APOLLO_ROLEPLAYING, ALIGNMENT_FAKING)
    # Random predictor + 200 samples should be 0.5 ± ~0.1
    assert abs(metrics.auroc - 0.5) < 0.1


def test_compute_metrics_raises_on_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="mismatched shapes"):
        compute_metrics(
            np.array([0, 1, 0]),
            np.array([0.1, 0.2]),
            APOLLO_ROLEPLAYING,
            ALIGNMENT_FAKING,
        )


def test_compute_metrics_raises_on_single_class() -> None:
    with pytest.raises(ValueError, match="both classes"):
        compute_metrics(
            np.array([1, 1, 1, 1]),
            np.array([0.1, 0.2, 0.3, 0.4]),
            APOLLO_ROLEPLAYING,
            ALIGNMENT_FAKING,
        )


def test_tpr_at_fpr_returns_value_in_unit_interval() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    tpr = tpr_at_fpr(labels, scores, target_fpr=0.5)
    assert 0.0 <= tpr <= 1.0


def test_tpr_at_fpr_at_full_budget_is_one() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    assert tpr_at_fpr(labels, scores, target_fpr=1.0) == 1.0


def test_tpr_at_fpr_on_perfect_predictions_is_one_even_at_zero() -> None:
    # With perfect separation, threshold between negatives and positives gives
    # FPR=0 and TPR=1, so the answer at any target_fpr (including 0.0) is 1.0.
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert tpr_at_fpr(labels, scores, target_fpr=0.0) == 1.0


def test_tpr_at_fpr_rejects_invalid_target() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    for bad in [-0.1, 1.1, 2.0]:
        with pytest.raises(ValueError, match="target_fpr"):
            tpr_at_fpr(labels, scores, target_fpr=bad)


def test_tpr_at_fpr_rejects_single_class_labels() -> None:
    with pytest.raises(ValueError, match="both classes"):
        tpr_at_fpr(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3]), target_fpr=0.5)


def test_plot_roc_curve_adds_line_and_baseline() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    fig, ax = plt.subplots()
    try:
        plot_roc_curve(labels, scores, label="apollo→AF", ax=ax)
        line_labels = [line.get_label() for line in ax.get_lines()]
        assert any("apollo→AF" in str(label) for label in line_labels)
        assert any("AUROC" in str(label) for label in line_labels)
        assert any(str(label) == "Random baseline" for label in line_labels)
    finally:
        plt.close(fig)


def test_plot_roc_curve_baseline_added_only_once() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    fig, ax = plt.subplots()
    try:
        plot_roc_curve(labels, scores, label="curve A", ax=ax)
        plot_roc_curve(labels, scores, label="curve B", ax=ax)
        baseline_count = sum(1 for line in ax.get_lines() if line.get_label() == "Random baseline")
        assert baseline_count == 1
    finally:
        plt.close(fig)
