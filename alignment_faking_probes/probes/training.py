"""Train and apply linear probes on activation datasets.

A ProbeResult bundles the fitted LogisticRegression with the StandardScaler
it was trained against, so the two can never be loaded separately and drift
out of sync. Loading a probe with the wrong scaler silently produces
incorrect predictions — saving them together as one object eliminates that
class of bug. All predictions go through ``predict_probe``, which always
uses the training scaler.

LOOCV AUROC is computed once at training time and stored as a
within-distribution sanity check. It is not the headline metric for the
transfer experiment — that comes from ``predict_probe`` applied to a
held-out scenario's activations.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from alignment_faking_probes.data.activation_store import ActivationDataset

#: Minimum positive count required for training. With only one positive,
#: LOOCV cannot estimate AUROC sensibly and the trained model is too
#: brittle to be worth saving. Fail fast rather than produce garbage.
_MIN_POSITIVES_FOR_TRAINING = 2


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
        train_auroc: LOOCV AUROC on the training set — within-distribution
            sanity check, NOT the transfer metric.
        n_train: Total number of training samples.
        n_positive: Number of positive (label=1) training samples.
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
    """Train a logistic regression probe with LOOCV AUROC reported.

    Fits a ``StandardScaler → LogisticRegression(class_weight='balanced')``
    pipeline on the full dataset, and computes LOOCV AUROC as a
    within-distribution sanity check using out-of-fold predictions.

    The pipeline is used for LOOCV so the scaler is refit per fold — this
    prevents the (small but real) leakage that would occur if the scaler
    were fit once on the full data and then used during LOO evaluation.
    The final fitted scaler + probe that get returned are fit on the full
    training set.

    Args:
        dataset: ActivationDataset containing activations, labels, scenario
            info, and other provenance.
        seed: Random seed passed to LogisticRegression for reproducibility.

    Returns:
        ProbeResult bundling the trained probe, fitted scaler, and metrics.

    Raises:
        ValueError: If the dataset has fewer than 2 positive examples.
    """
    if dataset.n_positive < _MIN_POSITIVES_FOR_TRAINING:
        raise ValueError(
            f"Need at least {_MIN_POSITIVES_FOR_TRAINING} positive examples to train, "
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

    loo = LeaveOneOut()
    oof_proba = cross_val_predict(
        pipeline,
        dataset.activations,
        dataset.labels,
        cv=loo,
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
        n_train=int(dataset.labels.shape[0]),
        n_positive=int(dataset.n_positive),
        metadata={
            "model_name": dataset.model_name,
            "seed": str(seed),
            **dataset.metadata,
        },
    )


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
