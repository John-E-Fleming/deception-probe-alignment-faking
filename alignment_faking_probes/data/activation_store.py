"""Save and load ActivationDataset objects.

ActivationDataset is the project's canonical serialisation format for
labelled activation tensors. One file per (scenario, layer) combination.
Activations, per-token labels, and a per-token sample_id are stored in a
compressed .npz; the rest of the dataclass is stored in a JSON sidecar
with the same stem. The two files are loaded together so labels and
sample membership can never drift out of sync with the activations they
describe.

Per-token shape contract: activations are stored at token granularity
(one row per token at a detection position) so probes can be trained
against Apollo Research's published per-token probe weights. The
``sample_id`` column maps each token back to its source sample; per-token
labels are sample-level labels broadcast across the sample's tokens.
``n_positive`` / ``n_negative`` are per-sample counts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from alignment_faking_probes import RECOGNISED_SCENARIOS

#: Sidecar JSON format version. Bumped whenever the on-disk schema changes.
#: v1 was the pre-Apollo-alignment per-sample format (no sample_id field);
#: v2 is the current per-token format with sample_id. Old v1 files are
#: rejected at load time — there is no in-place upgrade path.
SIDECAR_FORMAT_VERSION = "2"


@dataclass
class ActivationDataset:
    """Labelled per-token residual stream activations for a single (scenario, layer) pair.

    Attributes:
        activations: Float array of shape ``(n_tokens, hidden_dim)``.
        labels: Integer array of shape ``(n_tokens,)`` with values in ``{0, 1}``.
            Per-token labels are sample-level labels broadcast across the
            tokens of each sample; all tokens for a given ``sample_id`` must
            share the same label.
        sample_id: Integer array of shape ``(n_tokens,)`` mapping each token
            back to its source sample. IDs must be contiguous from 0 to
            ``n_samples - 1`` so downstream group-aware operations
            (GroupKFold, per-sample aggregation) can rely on dense indexing.
        scenario: One of the four constants in
            ``alignment_faking_probes.RECOGNISED_SCENARIOS``.
        layer: Residual stream layer index the activations were extracted from.
        model_name: Hugging Face model identifier the activations came from
            (e.g. ``"meta-llama/Llama-3.3-70B-Instruct"``).
        n_positive: Number of **samples** (not tokens) with label == 1.
        n_negative: Number of **samples** (not tokens) with label == 0.
        metadata: Free-form provenance dict. Conventional keys:
            ``apollo_commit``, ``extraction_date``, ``detection_mask_kind``,
            ``split`` ("train" | "test").

    Raises:
        ValueError: If ``scenario`` is not a recognised scenario name.
        ValueError: If ``activations``, ``labels``, or ``sample_id`` have
            mismatched lengths along axis 0.
        ValueError: If ``sample_id`` is not an integer array.
        ValueError: If ``sample_id`` values are not contiguous from 0.
        ValueError: If any sample has mixed labels (tokens within a sample
            disagree on their label).
        ValueError: If ``n_positive`` / ``n_negative`` do not match the
            per-sample label counts.
    """

    activations: np.ndarray
    labels: np.ndarray
    sample_id: np.ndarray
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

        n_tokens = self.activations.shape[0]
        if self.labels.shape != (n_tokens,):
            raise ValueError(
                f"activations and labels have mismatched lengths: "
                f"activations has {n_tokens} rows, labels has shape {self.labels.shape}"
            )
        if self.sample_id.shape != (n_tokens,):
            raise ValueError(
                f"activations and sample_id have mismatched lengths: "
                f"activations has {n_tokens} rows, sample_id has shape {self.sample_id.shape}"
            )
        if not np.issubdtype(self.sample_id.dtype, np.integer):
            raise ValueError(
                f"sample_id must be an integer array, got dtype {self.sample_id.dtype}"
            )

        unique_ids = np.unique(self.sample_id)
        n_samples = unique_ids.shape[0]
        if n_samples > 0 and (unique_ids[0] != 0 or unique_ids[-1] != n_samples - 1):
            raise ValueError(
                f"sample_id values must be contiguous integers from 0 to n_samples-1; "
                f"got {n_samples} unique ids ranging {int(unique_ids[0])}..{int(unique_ids[-1])}"
            )

        # Per-sample label consistency: every token within a sample must share
        # the same label. The broadcast happens at extraction time; a mismatch
        # here means someone hand-built a malformed dataset.
        sample_labels = np.empty(n_samples, dtype=self.labels.dtype)
        for sid in range(n_samples):
            mask = self.sample_id == sid
            sample_token_labels = self.labels[mask]
            first_label = sample_token_labels[0]
            if not np.all(sample_token_labels == first_label):
                raise ValueError(
                    f"Sample {sid} has mixed labels {np.unique(sample_token_labels).tolist()}; "
                    "all tokens within a sample must share a single label."
                )
            sample_labels[sid] = first_label

        actual_positive = int((sample_labels == 1).sum())
        actual_negative = int((sample_labels == 0).sum())
        if self.n_positive != actual_positive:
            raise ValueError(
                f"n_positive={self.n_positive} does not match per-sample label counts "
                f"(actual positive samples: {actual_positive})"
            )
        if self.n_negative != actual_negative:
            raise ValueError(
                f"n_negative={self.n_negative} does not match per-sample label counts "
                f"(actual negative samples: {actual_negative})"
            )


def save_activation_dataset(dataset: ActivationDataset, path: Path) -> None:
    """Save an ActivationDataset to disk.

    Writes a compressed .npz at ``path`` holding the activations, labels,
    and sample_id, and a JSON sidecar at ``path.with_suffix(".json")``
    holding the rest of the dataclass fields plus a ``format_version`` tag.

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
        sample_id=dataset.sample_id,
    )

    sidecar_path = path.with_suffix(".json")
    sidecar_payload: dict[str, object] = {
        "format_version": SIDECAR_FORMAT_VERSION,
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
        ValueError: If the sidecar ``format_version`` is not the current
            ``SIDECAR_FORMAT_VERSION`` (old per-sample v1 files are rejected
            — re-extract with the current pipeline rather than upgrading
            in place).
        ValueError: If the .npz is missing the ``sample_id`` array.
        ValueError: If the loaded payload fails
            ``ActivationDataset.__post_init__`` validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Activation .npz not found: {path}")

    sidecar_path = path.with_suffix(".json")
    if not sidecar_path.exists():
        raise FileNotFoundError(f"Activation JSON sidecar not found: {sidecar_path}")

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    format_version = sidecar.get("format_version", "1")
    if format_version != SIDECAR_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported activation dataset format_version={format_version!r}; "
            f"expected {SIDECAR_FORMAT_VERSION!r}. Re-extract activations with "
            "the current per-token pipeline rather than upgrading in place."
        )

    with np.load(path, allow_pickle=False) as npz:
        if "sample_id" not in npz.files:
            raise ValueError(
                f"Activation .npz at {path} is missing 'sample_id'; this looks "
                "like an old per-sample file. Re-extract."
            )
        activations = npz["activations"]
        labels = npz["labels"]
        sample_id = npz["sample_id"]

    return ActivationDataset(
        activations=activations,
        labels=labels,
        sample_id=sample_id,
        scenario=sidecar["scenario"],
        layer=sidecar["layer"],
        model_name=sidecar["model_name"],
        n_positive=sidecar["n_positive"],
        n_negative=sidecar["n_negative"],
        metadata=sidecar["metadata"],
    )
