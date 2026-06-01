"""Tests for probe evaluation metrics and ROC plotting."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from alignment_faking_probes import ALIGNMENT_FAKING, APOLLO_ROLEPLAYING
from alignment_faking_probes.probes.evaluation import (
    bootstrap_auroc_ci,
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


def test_bootstrap_auroc_ci_perfect_ranking_is_degenerate() -> None:
    """When every positive outscores every negative, every bootstrap resample
    also has perfect ranking — CI collapses to [1.0, 1.0]."""
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    lower, upper = bootstrap_auroc_ci(labels, scores, n_boot=500, seed=0)
    assert lower == 1.0
    assert upper == 1.0


def test_bootstrap_auroc_ci_inverted_ranking_is_degenerate() -> None:
    """When every negative outscores every positive, AUROC is 0 in every
    resample — CI collapses to [0.0, 0.0]."""
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
    lower, upper = bootstrap_auroc_ci(labels, scores, n_boot=500, seed=0)
    assert lower == 0.0
    assert upper == 0.0


def test_bootstrap_auroc_ci_contains_point_estimate() -> None:
    """For a non-degenerate case, the bootstrap CI should always contain the
    point estimate (it's a percentile of resampled AUROCs which mostly
    fluctuate around the population value)."""
    rng = np.random.default_rng(42)
    n_pos = 20
    n_neg = 50
    pos_scores = rng.normal(loc=0.7, scale=0.2, size=n_pos)
    neg_scores = rng.normal(loc=0.3, scale=0.2, size=n_neg)
    labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    scores = np.concatenate([pos_scores, neg_scores])

    from sklearn.metrics import roc_auc_score

    point = float(roc_auc_score(labels, scores))
    lower, upper = bootstrap_auroc_ci(labels, scores, n_boot=1000, seed=0)
    assert lower <= point <= upper
    # Sanity: CI shouldn't be implausibly narrow at this sample size
    assert (upper - lower) > 0.02


def test_bootstrap_auroc_ci_seed_is_reproducible() -> None:
    labels = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    scores = np.array([0.1, 0.3, 0.4, 0.6, 0.5, 0.8, 0.2, 0.7])
    ci1 = bootstrap_auroc_ci(labels, scores, n_boot=200, seed=123)
    ci2 = bootstrap_auroc_ci(labels, scores, n_boot=200, seed=123)
    assert ci1 == ci2


def test_bootstrap_auroc_ci_rejects_single_class() -> None:
    labels = np.array([1, 1, 1, 1])
    scores = np.array([0.1, 0.3, 0.5, 0.7])
    with pytest.raises(ValueError, match="both classes"):
        bootstrap_auroc_ci(labels, scores, n_boot=100)


def test_bootstrap_auroc_ci_rejects_non_positive_n_boot() -> None:
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.7, 0.3, 0.9])
    with pytest.raises(ValueError, match="n_boot must be positive"):
        bootstrap_auroc_ci(labels, scores, n_boot=0)


def test_compute_metrics_with_n_boot_populates_ci_fields() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    metrics = compute_metrics(
        labels,
        scores,
        APOLLO_ROLEPLAYING,
        ALIGNMENT_FAKING,
        n_boot=200,
        seed=0,
    )
    assert metrics.n_boot == 200
    assert metrics.auroc_ci is not None
    assert metrics.tpr_at_1pct_fpr_ci is not None
    assert metrics.tpr_at_5pct_fpr_ci is not None
    # Perfect ranking — bootstrap CI for AUROC should collapse to [1.0, 1.0]
    assert metrics.auroc_ci == (1.0, 1.0)


def test_compute_metrics_without_n_boot_leaves_ci_fields_none() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.3, 0.7, 0.9])
    metrics = compute_metrics(labels, scores, APOLLO_ROLEPLAYING, ALIGNMENT_FAKING)
    assert metrics.n_boot == 0
    assert metrics.auroc_ci is None
    assert metrics.tpr_at_1pct_fpr_ci is None
    assert metrics.tpr_at_5pct_fpr_ci is None
