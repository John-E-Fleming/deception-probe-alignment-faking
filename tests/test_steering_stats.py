"""Unit tests for the paired-McNemar statistics used in the steering plot.

The McNemar p-value is a published headline statistic for Blog 4, so the
math gets a test even though the script lives under ``scripts/dev/``.
Hand-computed reference values use the exact binomial form:

    p = 2 * sum_{i=0..min(b,c)} C(b+c, i) / 2**(b+c)

Reference values cross-checked against scipy.stats.binomtest in the
corresponding regime (not imported here to keep the test dependency-light).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# scripts/dev/ isn't a package, so import the module directly by path.
_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "dev" / "plot_steering_curve.py"
_spec = importlib.util.spec_from_file_location("plot_steering_curve", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
plot_steering_curve = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plot_steering_curve)


# --- _exact_mcnemar_pvalue ----------------------------------------------


def test_mcnemar_no_discordant_pairs_returns_one() -> None:
    assert plot_steering_curve._exact_mcnemar_pvalue(0, 0) == 1.0


def test_mcnemar_symmetric_under_swap() -> None:
    """The two-sided p-value should be the same when b and c are swapped."""
    assert plot_steering_curve._exact_mcnemar_pvalue(3, 1) == pytest.approx(
        plot_steering_curve._exact_mcnemar_pvalue(1, 3)
    )


def test_mcnemar_all_flips_in_one_direction() -> None:
    """b=10, c=0 -> p = 2 * (1/1024) = 0.001953125. The clearest signal
    we'd ever publish with N=20 paired transcripts and ~half flipping."""
    p = plot_steering_curve._exact_mcnemar_pvalue(10, 0)
    assert p == pytest.approx(2 / 1024)


def test_mcnemar_handcomputed_small_case() -> None:
    """b=3, c=1, n=4 discordant. Cumulative tail P(X<=1 | n=4, p=0.5)
    = (C(4,0) + C(4,1)) / 16 = 5/16. Two-sided = 10/16 = 0.625."""
    assert plot_steering_curve._exact_mcnemar_pvalue(3, 1) == pytest.approx(10 / 16)


def test_mcnemar_capped_at_one() -> None:
    """Equal split has tail >= 0.5 -- two-sided shouldn't exceed 1.0."""
    p = plot_steering_curve._exact_mcnemar_pvalue(5, 5)
    assert 0.0 <= p <= 1.0


# --- _build_paired_table ------------------------------------------------


def _row(cohort: str, idx: int, alpha: float, label: int, conf: float) -> dict:
    return {
        "cohort": cohort,
        "transcript_idx": idx,
        "alpha": alpha,
        "label": label,
        "classifier_confidence": conf,
    }


def test_paired_table_full_2x2_coverage() -> None:
    """4 transcripts -- one in each of the (baseline, steered) categories.

    idx=0: baseline NOT-AF, steered NOT-AF        -> a
    idx=1: baseline NOT-AF, steered AF            -> b
    idx=2: baseline AF,     steered NOT-AF        -> c
    idx=3: baseline AF,     steered AF            -> d
    """
    rows = [
        _row("af_partial", 0, 0.0, 0, 0.0),
        _row("af_partial", 0, 2.0, 0, 0.0),
        _row("af_partial", 1, 0.0, 0, 0.0),
        _row("af_partial", 1, 2.0, 1, 1.0),
        _row("af_partial", 2, 0.0, 1, 1.0),
        _row("af_partial", 2, 2.0, 0, 0.0),
        _row("af_partial", 3, 0.0, 1, 1.0),
        _row("af_partial", 3, 2.0, 1, 1.0),
    ]
    result = plot_steering_curve._build_paired_table(
        rows, cohort="af_partial", baseline_alpha=0.0, steered_alpha=2.0
    )
    assert result is not None
    a, b, c, d, paired_idx = result
    assert (a, b, c, d) == (1, 1, 1, 1)
    assert paired_idx == [0, 1, 2, 3]


def test_paired_table_drops_unpaired_transcripts() -> None:
    """Transcripts present at only one alpha shouldn't appear in the table."""
    rows = [
        _row("af_partial", 0, 0.0, 0, 0.0),
        _row("af_partial", 0, 2.0, 1, 1.0),
        # idx=1 only has a baseline row
        _row("af_partial", 1, 0.0, 0, 0.0),
        # idx=2 only has a steered row
        _row("af_partial", 2, 2.0, 1, 1.0),
    ]
    result = plot_steering_curve._build_paired_table(
        rows, cohort="af_partial", baseline_alpha=0.0, steered_alpha=2.0
    )
    assert result is not None
    a, b, c, d, paired_idx = result
    assert (a, b, c, d) == (0, 1, 0, 0)
    assert paired_idx == [0]


def test_paired_table_returns_none_when_alpha_missing() -> None:
    """If baseline alpha is absent from the data, return None."""
    rows = [
        _row("af_partial", 0, 2.0, 1, 1.0),
        _row("af_partial", 1, 2.0, 1, 1.0),
    ]
    result = plot_steering_curve._build_paired_table(
        rows, cohort="af_partial", baseline_alpha=0.0, steered_alpha=2.0
    )
    assert result is None


def test_paired_table_respects_canonical_af_threshold() -> None:
    """label=1 with confidence < 0.6 is NOT canonical-AF (lower-bar 'yes'
    that doesn't clear the 3-of-5 criteria gate)."""
    rows = [
        _row("af_partial", 0, 0.0, 0, 0.0),
        # label=1 but confidence=0.4 -> below threshold, NOT canonical-AF
        _row("af_partial", 0, 2.0, 1, 0.4),
    ]
    result = plot_steering_curve._build_paired_table(
        rows, cohort="af_partial", baseline_alpha=0.0, steered_alpha=2.0
    )
    assert result is not None
    a, b, c, d, _ = result
    assert (a, b, c, d) == (1, 0, 0, 0)


def test_paired_table_filters_by_cohort() -> None:
    """A row from a different cohort must not leak into the table."""
    rows = [
        _row("af_partial", 0, 0.0, 0, 0.0),
        _row("af_partial", 0, 2.0, 1, 1.0),
        # Same transcript_idx in a different cohort -- must be ignored
        _row("af_control", 0, 0.0, 1, 1.0),
        _row("af_control", 0, 2.0, 1, 1.0),
    ]
    result = plot_steering_curve._build_paired_table(
        rows, cohort="af_partial", baseline_alpha=0.0, steered_alpha=2.0
    )
    assert result is not None
    a, b, c, d, _ = result
    assert (a, b, c, d) == (0, 1, 0, 0)


# --- _bootstrap_mean_ci -------------------------------------------------


def test_bootstrap_ci_empty_returns_zero() -> None:
    import numpy as np

    lo, hi = plot_steering_curve._bootstrap_mean_ci(np.array([], dtype=np.float64))
    assert (lo, hi) == (0.0, 0.0)


def test_bootstrap_ci_brackets_the_mean() -> None:
    """For any non-degenerate sample the bootstrap CI should contain
    the sample mean (the bootstrap distribution is centred on it)."""
    import numpy as np

    rng = np.random.default_rng(42)
    values = rng.normal(loc=100.0, scale=10.0, size=50)
    lo, hi = plot_steering_curve._bootstrap_mean_ci(values, n_boot=2000, rng_seed=1)
    assert lo < float(values.mean()) < hi


def test_bootstrap_ci_is_deterministic_given_seed() -> None:
    """Same seed + same input -> identical CI. Required so figure
    regeneration is reproducible."""
    import numpy as np

    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    a = plot_steering_curve._bootstrap_mean_ci(values, n_boot=500, rng_seed=7)
    b = plot_steering_curve._bootstrap_mean_ci(values, n_boot=500, rng_seed=7)
    assert a == b


# --- _spearman_rho_pvalue ----------------------------------------------


def test_spearman_perfect_positive_monotonic() -> None:
    import numpy as np

    x = np.arange(10, dtype=np.float64)
    y = 2 * x + 3  # strictly monotonic increasing
    rho, p_val = plot_steering_curve._spearman_rho_pvalue(x, y)
    assert rho == pytest.approx(1.0)
    # p-value for perfect rank correlation: the function returns 0.0
    # via the |rho|>=1 short-circuit
    assert p_val == pytest.approx(0.0)


def test_spearman_perfect_negative_monotonic() -> None:
    import numpy as np

    x = np.arange(10, dtype=np.float64)
    y = -x  # strictly monotonic decreasing
    rho, p_val = plot_steering_curve._spearman_rho_pvalue(x, y)
    assert rho == pytest.approx(-1.0)
    assert p_val == pytest.approx(0.0)


def test_spearman_small_n_returns_neutral() -> None:
    import numpy as np

    x = np.array([1.0, 2.0])
    y = np.array([3.0, 4.0])
    rho, p_val = plot_steering_curve._spearman_rho_pvalue(x, y)
    assert (rho, p_val) == (0.0, 1.0)


def test_spearman_constant_returns_neutral() -> None:
    """One of the inputs constant -> std==0 -> can't compute rho.
    Should return the neutral default."""
    import numpy as np

    x = np.arange(5, dtype=np.float64)
    y = np.full(5, 7.0)
    rho, p_val = plot_steering_curve._spearman_rho_pvalue(x, y)
    assert (rho, p_val) == (0.0, 1.0)


def test_spearman_pvalue_drops_with_sample_size() -> None:
    """Same correlation strength, larger sample -> smaller p-value.
    Sanity check on the t-distribution approximation."""
    import numpy as np

    rng = np.random.default_rng(11)
    # Create rank-correlated data with rho ~ 0.5
    x_small = rng.normal(size=20)
    y_small = x_small + rng.normal(scale=1.5, size=20)
    x_big = rng.normal(size=200)
    y_big = x_big + rng.normal(scale=1.5, size=200)
    _, p_small = plot_steering_curve._spearman_rho_pvalue(x_small, y_small)
    _, p_big = plot_steering_curve._spearman_rho_pvalue(x_big, y_big)
    assert p_big < p_small
