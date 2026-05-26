"""Train and apply linear probes on per-token activation datasets.

A ProbeResult bundles the fitted LogisticRegression with the StandardScaler
it was trained against, so the two can never be loaded separately and drift
out of sync. Loading a probe with the wrong scaler silently produces
incorrect predictions — saving them together as one object eliminates that
class of bug. All predictions go through ``predict_probe``, which always
uses the training scaler.

Training runs on per-token activations grouped by ``sample_id``. Group-aware
``GroupKFold`` CV ensures that no two tokens from the same sample ever
split across train and eval folds — without that grouping, AUROC would
be inflated by within-sample memorisation (tokens from the same source
dialogue are nearly identical). The per-token grouped-CV AUROC is stored
as ``ProbeResult.train_auroc`` and is a within-distribution sanity check,
not the transfer metric. Per-sample probe scores (used for the transfer
matrix) come from ``aggregate_token_scores`` applied to
``predict_probe`` output.
"""

from __future__ import annotations

import pickle
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from alignment_faking_probes.data.activation_store import ActivationDataset

#: Minimum positive **sample** count required for training. Two positives is
#: the floor: with only one, GroupKFold has nothing to leave out for the
#: held-out fold containing the positive sample, and the trained model is
#: too brittle to be worth saving. Fail fast rather than produce garbage.
_MIN_POSITIVES_FOR_TRAINING = 2

#: Default number of CV folds. The actual number used is
#: ``min(_DEFAULT_CV_SPLITS, n_samples)`` so tiny datasets degrade
#: gracefully.
_DEFAULT_CV_SPLITS = 5

#: Aggregation methods supported by :func:`aggregate_token_scores`.
AggregationMethod = Literal["mean", "max", "last"]
_VALID_AGGREGATION_METHODS = ("mean", "max", "last")


@dataclass
class ProbeResult:
    """A fitted linear probe and the scaler used to standardise its inputs.

    Attributes:
        probe: The fitted LogisticRegression.
        scaler: The StandardScaler that was fit on the same activations as
            the probe. Always travels with the probe — never load one
            without the other.
        layer: Residual stream layer index the training activations came from.
        train_scenario: Which scenario the probe was trained on (one of the
            constants in ``alignment_faking_probes.RECOGNISED_SCENARIOS``).
        train_auroc: Per-token GroupKFold-CV AUROC on the training set —
            within-distribution sanity check, NOT the transfer metric.
            Groups are sample IDs so no two tokens from the same sample
            split across folds.
        n_train: Number of **samples** in the training set (counted from
            unique sample IDs, not tokens).
        n_positive: Number of positive (label=1) **samples** in training.
        metadata: Free-form provenance dict. Conventional keys:
            ``apollo_commit``, ``model_name``, ``seed``.
    """

    probe: LogisticRegression
    scaler: StandardScaler
    layer: int
    train_scenario: str
    train_auroc: float
    n_train: int
    n_positive: int
    metadata: dict[str, str] = field(default_factory=dict)


def train_probe(dataset: ActivationDataset, seed: int) -> ProbeResult:
    """Train a logistic regression probe with grouped per-token CV AUROC.

    Fits a ``StandardScaler → LogisticRegression(class_weight='balanced')``
    pipeline on per-token activations and computes group-aware CV AUROC
    as a within-distribution sanity check.

    ``GroupKFold`` groups by ``dataset.sample_id`` so no two tokens from the
    same sample ever split across train and eval folds — without this,
    tokens from the same source dialogue (which are nearly identical at
    the residual-stream level) would leak across folds and inflate the
    reported AUROC. Per-fold scaler refitting (via the Pipeline) prevents
    the smaller scaler-fit leakage. The final fitted scaler + probe that
    get returned are fit on the full training set.

    Args:
        dataset: Per-token ActivationDataset with ``sample_id`` populated.
        seed: Random seed passed to LogisticRegression for reproducibility.

    Returns:
        ProbeResult bundling the trained probe, fitted scaler, and metrics.

    Raises:
        ValueError: If the dataset has fewer than 2 positive **samples**.
    """
    if dataset.n_positive < _MIN_POSITIVES_FOR_TRAINING:
        raise ValueError(
            f"Need at least {_MIN_POSITIVES_FOR_TRAINING} positive samples to train, "
            f"got {dataset.n_positive}"
        )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "probe",
                LogisticRegression(class_weight="balanced", random_state=seed, max_iter=1000),
            ),
        ]
    )

    n_samples = int(np.unique(dataset.sample_id).size)
    n_splits = min(_DEFAULT_CV_SPLITS, n_samples)
    cv = GroupKFold(n_splits=n_splits)
    oof_proba = cross_val_predict(
        pipeline,
        dataset.activations,
        dataset.labels,
        cv=cv,
        groups=dataset.sample_id,
        method="predict_proba",
    )[:, 1]
    train_auroc = float(roc_auc_score(dataset.labels, oof_proba))

    pipeline.fit(dataset.activations, dataset.labels)

    return ProbeResult(
        probe=pipeline.named_steps["probe"],
        scaler=pipeline.named_steps["scaler"],
        layer=dataset.layer,
        train_scenario=dataset.scenario,
        train_auroc=train_auroc,
        n_train=n_samples,
        n_positive=int(dataset.n_positive),
        metadata={
            "model_name": dataset.model_name,
            "seed": str(seed),
            **dataset.metadata,
        },
    )


def _iter_unique_ordered(sample_id: np.ndarray) -> Iterator[int]:
    """Yield contiguous 0..n-1 sample ids. Cheap helper for clarity at call sites."""
    n_samples = int(sample_id.max() + 1) if sample_id.size > 0 else 0
    yield from range(n_samples)


def aggregate_token_scores(
    scores: np.ndarray,
    sample_id: np.ndarray,
    method: AggregationMethod = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate per-token probe scores into per-sample scores.

    Apollo's ``prompt_scorer.py`` uses mean as its default; we match that
    for the headline transfer matrix and run max/last as ablations.

    Args:
        scores: Per-token scores of shape ``(n_tokens,)``.
        sample_id: Per-token sample identifiers, same shape as ``scores``.
            Must be contiguous integers from 0 to n_samples-1.
        method: How to combine tokens within a sample.

            - ``"mean"`` (default): average of token scores. Matches Apollo
              and gives a calibrated probability.
            - ``"max"``: maximum token score. Sensitive to outliers — useful
              for "is there *any* deceptive token?" framing.
            - ``"last"``: the score of the last token (by row index) in a
              sample. Useful when token ordering is meaningful.

    Returns:
        Tuple ``(per_sample_scores, sample_ids)`` where ``sample_ids`` is
        the canonical ordering ``0..n_samples-1`` used by ``per_sample_scores``.

    Raises:
        ValueError: If ``scores`` and ``sample_id`` have different shapes.
        ValueError: If ``method`` is not one of mean / max / last.
    """
    if scores.shape != sample_id.shape:
        raise ValueError(
            f"scores and sample_id must have the same shape; "
            f"got {scores.shape} and {sample_id.shape}"
        )
    if method not in _VALID_AGGREGATION_METHODS:
        raise ValueError(
            f"Unknown aggregation method {method!r}; expected one of {_VALID_AGGREGATION_METHODS}"
        )

    n_samples = int(sample_id.max() + 1) if sample_id.size > 0 else 0
    sample_ids = np.arange(n_samples, dtype=sample_id.dtype)
    per_sample = np.empty(n_samples, dtype=np.float64)

    if method == "mean":
        # Vectorised — sum / count per group.
        sums = np.bincount(sample_id, weights=scores, minlength=n_samples)
        counts = np.bincount(sample_id, minlength=n_samples)
        per_sample = sums / counts
    elif method == "max":
        for sid in _iter_unique_ordered(sample_id):
            per_sample[sid] = float(scores[sample_id == sid].max())
    else:  # "last"
        for sid in _iter_unique_ordered(sample_id):
            per_sample[sid] = float(scores[sample_id == sid][-1])

    return per_sample, sample_ids


def predict_probe(probe_result: ProbeResult, activations: np.ndarray) -> np.ndarray:
    """Apply a trained probe to new activations.

    Scales the activations using the training scaler bundled in
    ``probe_result`` — never fits a new scaler. This is the only correct way
    to apply a probe to held-out data; a freshly fit scaler on test data
    would discard the calibration learned during training and silently
    produce wrong scores.

    Args:
        probe_result: A trained ProbeResult.
        activations: Array of shape (n_samples, hidden_dim). ``hidden_dim``
            must match what the probe was trained on.

    Returns:
        Array of shape (n_samples,) with deception probability scores in [0, 1].

    Raises:
        ValueError: If ``activations.shape[1]`` does not match the probe's
            input feature dimension.
    """
    n_features_trained = probe_result.probe.n_features_in_
    if activations.shape[1] != n_features_trained:
        raise ValueError(
            f"Expected {n_features_trained} features per sample, got {activations.shape[1]}"
        )
    scaled = probe_result.scaler.transform(activations)
    return np.asarray(probe_result.probe.predict_proba(scaled)[:, 1])


def save_probe(probe_result: ProbeResult, path: Path) -> None:
    """Save a ProbeResult as a single .pkl file.

    Probe and scaler are persisted together; they are never split into
    separate files in this codebase.

    Args:
        probe_result: The fitted ProbeResult to save.
        path: Output path. Parent directory must already exist.

    Raises:
        FileNotFoundError: If ``path.parent`` does not exist.
    """
    path = Path(path)
    if not path.parent.exists():
        raise FileNotFoundError(f"Parent directory does not exist: {path.parent}")
    with path.open("wb") as f:
        pickle.dump(probe_result, f)


def load_probe(path: Path) -> ProbeResult:
    """Load a ProbeResult from disk.

    Only load .pkl files from trusted sources — pickle deserialisation
    executes arbitrary code by design.

    Args:
        path: Path to a .pkl file written by ``save_probe``.

    Returns:
        The deserialised ProbeResult with probe and scaler bundled.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the loaded object is not a ProbeResult.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Probe .pkl not found: {path}")
    with path.open("rb") as f:
        result = pickle.load(f)
    if not isinstance(result, ProbeResult):
        raise ValueError(f"Loaded object is not a ProbeResult: got {type(result).__name__}")
    return result
