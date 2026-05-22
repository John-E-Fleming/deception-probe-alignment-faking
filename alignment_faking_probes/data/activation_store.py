"""Save and load ActivationDataset objects.

ActivationDataset is the project's canonical serialisation format for
labelled activation tensors. One file per (scenario, layer) combination.
Activations and labels are stored as a compressed .npz; the rest of the
dataclass is stored in a JSON sidecar with the same stem. The two files are
loaded together so labels can never drift out of sync with the activations
they describe.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from alignment_faking_probes import RECOGNISED_SCENARIOS


@dataclass
class ActivationDataset:
    """Labelled residual stream activations for a single (scenario, layer) pair.

    Attributes:
        activations: Float array of shape (n_samples, hidden_dim).
        labels: Integer array of shape (n_samples,) containing values in {0, 1}.
        scenario: One of the four constants in
            ``alignment_faking_probes.RECOGNISED_SCENARIOS``.
        layer: Residual stream layer index the activations were extracted from.
        model_name: Hugging Face model identifier the activations came from
            (e.g. ``"meta-llama/Llama-3.3-70B-Instruct"``).
        n_positive: Number of samples with label == 1. Must equal
            ``int((labels == 1).sum())``.
        n_negative: Number of samples with label == 0. Must equal
            ``int((labels == 0).sum())``.
        metadata: Free-form provenance dict. Conventional keys:
            ``apollo_commit``, ``extraction_date``, ``quantisation``,
            ``split`` ("train" | "test").

    Raises:
        ValueError: If ``scenario`` is not a recognised scenario name.
        ValueError: If ``activations`` and ``labels`` have different lengths.
        ValueError: If ``n_positive`` or ``n_negative`` does not match the
            counts computed from ``labels``.
    """

    activations: np.ndarray
    labels: np.ndarray
    scenario: str
    layer: int
    model_name: str
    n_positive: int
    n_negative: int
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.scenario not in RECOGNISED_SCENARIOS:
            raise ValueError(
                f"Unrecognised scenario {self.scenario!r}; expected one of {RECOGNISED_SCENARIOS}"
            )
        if self.activations.shape[0] != self.labels.shape[0]:
            raise ValueError(
                f"activations and labels have mismatched lengths: "
                f"{self.activations.shape[0]} vs {self.labels.shape[0]}"
            )
        actual_positive = int((self.labels == 1).sum())
        actual_negative = int((self.labels == 0).sum())
        if self.n_positive != actual_positive:
            raise ValueError(
                f"n_positive={self.n_positive} does not match label counts "
                f"(actual positive: {actual_positive})"
            )
        if self.n_negative != actual_negative:
            raise ValueError(
                f"n_negative={self.n_negative} does not match label counts "
                f"(actual negative: {actual_negative})"
            )


def save_activation_dataset(dataset: ActivationDataset, path: Path) -> None:
    """Save an ActivationDataset to disk.

    Writes a compressed .npz at ``path`` holding the activations and labels,
    and a JSON sidecar at ``path.with_suffix(".json")`` holding the rest of
    the dataclass fields.

    The .npz extension is enforced — if ``path`` lacks it, ``.npz`` is added.

    Args:
        dataset: The dataset to save. Validation lives in
            ``ActivationDataset.__post_init__`` and has already run by the
            time you reach this function.
        path: Path for the .npz file. The sidecar path is derived from it.

    Raises:
        FileNotFoundError: If ``path.parent`` does not exist.
    """
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    if not path.parent.exists():
        raise FileNotFoundError(f"Parent directory does not exist: {path.parent}")

    np.savez_compressed(
        path,
        activations=dataset.activations,
        labels=dataset.labels,
    )

    sidecar_path = path.with_suffix(".json")
    sidecar_payload: dict[str, object] = {
        "scenario": dataset.scenario,
        "layer": dataset.layer,
        "model_name": dataset.model_name,
        "n_positive": dataset.n_positive,
        "n_negative": dataset.n_negative,
        "metadata": dataset.metadata,
    }
    sidecar_path.write_text(json.dumps(sidecar_payload, indent=2), encoding="utf-8")


def load_activation_dataset(path: Path) -> ActivationDataset:
    """Load an ActivationDataset previously written by ``save_activation_dataset``.

    Args:
        path: Path to the .npz file. The JSON sidecar must exist alongside it.

    Returns:
        ActivationDataset with all fields restored.

    Raises:
        FileNotFoundError: If the .npz file or its JSON sidecar is missing.
        ValueError: If the loaded payload fails
            ``ActivationDataset.__post_init__`` validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Activation .npz not found: {path}")

    sidecar_path = path.with_suffix(".json")
    if not sidecar_path.exists():
        raise FileNotFoundError(f"Activation JSON sidecar not found: {sidecar_path}")

    with np.load(path, allow_pickle=False) as npz:
        activations = npz["activations"]
        labels = npz["labels"]

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

    return ActivationDataset(
        activations=activations,
        labels=labels,
        scenario=sidecar["scenario"],
        layer=sidecar["layer"],
        model_name=sidecar["model_name"],
        n_positive=sidecar["n_positive"],
        n_negative=sidecar["n_negative"],
        metadata=sidecar["metadata"],
    )
