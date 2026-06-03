"""Per-token residual stream activation extraction via Apollo's pipeline.

We delegate the actual model forward pass and detection masking to Apollo
Research's ``deception_detection.activations.Activations.from_model``, then
**convert** Apollo's per-token output into our own ``ActivationDataset``
format (``.npz`` + JSON sidecar, indexed by ``sample_id``). Delegating the
forward pass is what makes our activations directly compatible with
Apollo's published probe weights — the alternative of running our own
NNSight code introduced silent format mismatches.

Two layers of code:

1. **Pure helpers** (``_af_transcript_to_dialogue_payload``,
   ``_build_activation_datasets_from_apollo_output``) operate on
   plain-Python / NumPy inputs only. No Apollo or torch imports needed.
   Fully unit-tested locally without a GPU.

2. **Apollo-coupled wrappers** (``build_af_dialogue_dataset``,
   ``extract_activations``) call into Apollo's ``Dialogue`` /
   ``DialogueDataset`` / ``Activations`` types. Validated on RunPod via
   ``scripts/smoke_test_extraction.py``; the Apollo API surface may need
   small adjustments depending on the pinned Apollo commit (see
   ``pyproject.toml`` for the pin).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from alignment_faking_probes.data.activation_store import ActivationDataset
from alignment_faking_probes.data.labeling import LabelledTranscript

if TYPE_CHECKING:
    from deception_detection.data.base import DialogueDataset
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

#: Valid detection-mask scopes for alignment-faking transcripts.
#: - ``"scratchpad"``: mark only the model's scratchpad reasoning tokens.
#: - ``"response"``: mark only the model's final response tokens (matches
#:   Apollo's choice for their scenarios).
#: - ``"both"`` (default for AF headline): mark scratchpad + response.
DetectionMaskKind = Literal["scratchpad", "response", "both"]
_VALID_DETECTION_MASK_KINDS = ("scratchpad", "response", "both")


@dataclass
class ExtractionConfig:
    """Configuration for residual-stream activation extraction.

    Apollo's ``Activations.from_model`` loads the model with its own
    precision / autocast handling, so quantisation no longer lives in this
    config — the caller passes a preloaded model.

    Attributes:
        model_name: Hugging Face model identifier (recorded in metadata for
            traceability; the actual model is passed in separately).
        layers: Residual stream layer indices to extract.
        batch_size: Forward-pass batch size handed to ``Activations.from_model``.
        max_length: Maximum token length; longer inputs are truncated.
        detection_mask_kind: Which tokens count as "detection positions"
            for AF transcripts. Apollo scenarios bring their own masks via
            their ``DialogueDataset``; this setting is only consulted by
            ``build_af_dialogue_dataset``.

    Raises:
        ValueError: If any field is out of its valid range.
    """

    model_name: str
    layers: list[int]
    batch_size: int
    max_length: int
    detection_mask_kind: DetectionMaskKind = "both"

    def __post_init__(self) -> None:
        if self.detection_mask_kind not in _VALID_DETECTION_MASK_KINDS:
            raise ValueError(
                f"detection_mask_kind must be one of {_VALID_DETECTION_MASK_KINDS}, "
                f"got {self.detection_mask_kind!r}"
            )
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")
        if not self.layers:
            raise ValueError("layers must be non-empty")
        if any(layer < 0 for layer in self.layers):
            raise ValueError(f"All layers must be non-negative, got {self.layers}")


# ---------------------------------------------------------------------------
# Pure helpers — no Apollo / torch imports. Fully unit-tested locally.
# ---------------------------------------------------------------------------


def _check_all_have_scratchpad(labelled: list[LabelledTranscript]) -> None:
    """Raise ``ValueError`` if any transcript has ``scratchpad is None``.

    Atomic pre-check used by ``build_af_dialogue_dataset`` so the failure
    mode reports the count of bad items and the first few indices in one
    error rather than dying mid-comprehension on the first bad item.
    """
    bad_idx = [i for i, item in enumerate(labelled) if item.transcript.scratchpad is None]
    if not bad_idx:
        return
    preview = bad_idx[:5]
    raise ValueError(
        f"build_af_dialogue_dataset: {len(bad_idx)} of {len(labelled)} transcripts "
        f"have scratchpad=None (first indices: {preview}"
        f"{'...' if len(bad_idx) > 5 else ''}). "
        "Drop them with filter_included() or fix the upstream filter."
    )


def _check_detection_mask_nonempty(mask: np.ndarray) -> None:
    """Raise ``ValueError`` if the tokenised detection mask has zero True positions.

    Used by ``extract_activations`` to fail fast before the forward
    pass. An all-False mask means downstream extraction would return
    zero token rows and ``ActivationDataset`` construction would then
    crash with a confusing per-sample-label shape mismatch.

    Args:
        mask: Boolean array of shape ``(n_dialogues, seq_len)``.
    """
    n_true = int(mask.sum())
    if n_true > 0:
        return
    n_dialogues = mask.shape[0] if mask.ndim >= 1 else 0
    raise ValueError(
        f"extract_activations: tokenised detection_mask has zero True positions "
        f"across all {n_dialogues} dialogues. The forward pass would extract zero "
        "tokens. Most likely cause: the DialogueDataset's messages don't have "
        "any `detect=True` flags (Apollo prewritten variants without paired "
        "detection positions, or a malformed build_af_dialogue_dataset input). "
        "Fix the input dataset rather than running extraction."
    )


def _af_transcript_to_dialogue_payload(
    labelled: LabelledTranscript,
    mask_kind: DetectionMaskKind,
) -> dict[str, Any]:
    """Build the message + detection-mask payload for one AF transcript.

    Returns a plain dict matching the kwargs Apollo's ``Dialogue``
    constructor expects. Keeping this pure-Python means we can unit-test
    the message construction and mask logic locally; the Apollo integration
    point lives in ``build_af_dialogue_dataset``.

    The returned dict has:

    - ``messages``: list of ``{role, content, detect}`` dicts. ``detect`` is
      a per-message bool that flags whether the message's tokens should be
      included in the detection mask (per the chosen ``mask_kind``).
    - ``label``: ``"deceptive"`` if ``labelled.label == 1`` else ``"honest"``
      (Apollo's convention).

    Args:
        labelled: A LabelledTranscript with a non-None scratchpad.
        mask_kind: Which tokens to flag for the detection mask.

    Raises:
        ValueError: If ``labelled.transcript.scratchpad`` is None (callers
            should ``filter_included`` first).
    """
    if labelled.transcript.scratchpad is None:
        raise ValueError(
            "Transcript with no scratchpad encountered; "
            "call filter_included first to drop excluded transcripts."
        )

    detect_scratchpad = mask_kind in ("scratchpad", "both")
    detect_response = mask_kind in ("response", "both")

    # Apollo's Role is Literal["system", "user", "assistant"] only — no
    # dedicated scratchpad role. The Message docstring notes that adjacent
    # same-role messages get concatenated, so we represent the scratchpad
    # and the final response as two consecutive `assistant` messages with
    # independent `detect` flags. The flag-per-message lets the chosen
    # mask_kind cover scratchpad-only, response-only, or both.
    return {
        "messages": [
            {"role": "user", "content": labelled.transcript.prompt, "detect": False},
            {
                "role": "assistant",
                "content": labelled.transcript.scratchpad,
                "detect": detect_scratchpad,
            },
            {
                "role": "assistant",
                "content": labelled.transcript.response,
                "detect": detect_response,
            },
        ],
        "label": "deceptive" if labelled.label == 1 else "honest",
    }


def _build_activation_datasets_from_apollo_output(
    masked_acts: np.ndarray,
    sample_id: np.ndarray,
    sample_labels: np.ndarray,
    config: ExtractionConfig,
    scenario: str,
    apollo_commit: str,
) -> dict[int, ActivationDataset]:
    """Convert Apollo's per-token activations into our ActivationDataset format.

    Apollo's ``Activations.get_masked_activations()`` returns
    ``(n_tokens, n_layers, hidden_dim)``. We slice out each requested layer
    and pair it with per-token labels (sample-level labels broadcast across
    each sample's tokens).

    Args:
        masked_acts: Apollo's flattened activations, shape
            ``(n_tokens, n_layers, hidden_dim)``. Layer axis order MUST
            match ``config.layers``.
        sample_id: Per-token sample identifiers of shape ``(n_tokens,)``.
        sample_labels: Per-sample binary labels of shape ``(n_samples,)``.
            Broadcast across tokens via ``sample_id``.
        config: ExtractionConfig used to label the output (only ``model_name``,
            ``layers``, and ``detection_mask_kind`` are read here).
        scenario: Scenario name string for the output ActivationDatasets.
        apollo_commit: Apollo repo commit hash for provenance metadata.

    Returns:
        Dict mapping layer index → ActivationDataset.

    Raises:
        ValueError: If ``masked_acts``'s layer axis doesn't match ``len(config.layers)``.
        ValueError: If ``sample_id`` and ``masked_acts`` row counts disagree.
        ValueError: If ``sample_labels`` length doesn't match the unique sample count.
    """
    n_tokens = masked_acts.shape[0]
    if sample_id.shape != (n_tokens,):
        raise ValueError(
            f"masked_acts rows ({n_tokens}) and sample_id ({sample_id.shape}) mismatch"
        )
    if masked_acts.shape[1] != len(config.layers):
        raise ValueError(
            f"masked_acts has {masked_acts.shape[1]} layers but config requested "
            f"{len(config.layers)} ({config.layers})"
        )
    unique_ids = np.unique(sample_id)
    n_samples = unique_ids.shape[0]
    if sample_labels.shape != (n_samples,):
        raise ValueError(f"sample_labels shape {sample_labels.shape} != expected ({n_samples},)")

    # Broadcast sample-level labels to per-token labels via sample_id.
    per_token_labels = sample_labels[sample_id]
    n_positive = int((sample_labels == 1).sum())
    n_negative = int((sample_labels == 0).sum())

    metadata = {
        "apollo_commit": apollo_commit,
        "detection_mask_kind": config.detection_mask_kind,
        "extraction_date": datetime.now(timezone.utc).isoformat(),
    }

    datasets: dict[int, ActivationDataset] = {}
    for layer_axis_idx, layer in enumerate(config.layers):
        layer_acts = np.ascontiguousarray(masked_acts[:, layer_axis_idx, :])
        datasets[layer] = ActivationDataset(
            activations=layer_acts.astype(np.float32, copy=False),
            labels=per_token_labels.astype(np.int64, copy=False),
            sample_id=sample_id.astype(np.int64, copy=False),
            scenario=scenario,
            layer=layer,
            model_name=config.model_name,
            n_positive=n_positive,
            n_negative=n_negative,
            metadata=metadata.copy(),
        )
    return datasets


# ---------------------------------------------------------------------------
# Apollo-coupled wrappers — exercised on RunPod via the smoke test.
# ---------------------------------------------------------------------------


def build_af_dialogue_dataset(
    labelled: list[LabelledTranscript],
    mask_kind: DetectionMaskKind,
    shuffle_upon_init: bool = False,
) -> "DialogueDataset":
    """Wrap AF transcripts as an Apollo DialogueDataset.

    Apollo's ``DialogueDataset.__init__`` accepts ``dialogues`` and
    ``labels`` directly when subclasses want to bypass ``_get_dialogues``.
    We pass ``skip_variant_validation=True`` because the AF data isn't a
    registered variant of the base class.

    Args:
        labelled: AF transcripts to convert. Must all have ``included=True``
            and a non-None ``scratchpad``.
        mask_kind: Which tokens count as detection positions (see
            ``DetectionMaskKind``).

    Returns:
        An Apollo ``DialogueDataset`` ready to pass to ``extract_activations``.

    Raises:
        ValueError: If any transcript has ``scratchpad is None``. The
            check runs once over the full list at entry rather than per
            item, so the failure mode reports the count of bad items
            and the first few indices rather than dying midway through
            the comprehension below.
    """
    _check_all_have_scratchpad(labelled)

    from deception_detection.data.base import DialogueDataset
    from deception_detection.types import Label, Message

    payloads = [_af_transcript_to_dialogue_payload(item, mask_kind) for item in labelled]
    dialogues: list[list[Message]] = []
    labels: list[Label] = []
    for payload in payloads:
        dialogues.append([Message(**msg) for msg in payload["messages"]])
        labels.append(Label.DECEPTIVE if payload["label"] == "deceptive" else Label.HONEST)

    # shuffle_upon_init=False is CRUCIAL. Apollo's DialogueDataset.__init__
    # defaults to shuffling dialogues with np.random.seed(42) whenever
    # mixed labels are present (deception_detection/data/base.py lines
    # 152-155). The shuffle is internally consistent (dialogues + labels
    # reorder together), so AUROC results are correct -- but it makes
    # post-hoc indexing into our input `labelled` list mismatch with
    # `sample_id`: sample_id=i refers to shuffled position i, not input
    # position i. That broke our per-token inspection by attaching the
    # wrong AdvBench prompts to the right activations. Disabling shuffle
    # keeps input order == tokenised order == sample_id order, which is
    # the contract our inspect_top_tokens.py and downstream scripts assume.
    return DialogueDataset(
        dialogues=dialogues,
        labels=labels,
        skip_variant_validation=True,
        shuffle_upon_init=shuffle_upon_init,
    )


def extract_activations(
    dialogue_dataset: "DialogueDataset",
    config: ExtractionConfig,
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    scenario: str,
    apollo_commit: str,
) -> dict[int, ActivationDataset]:
    """Extract per-token residual stream activations using Apollo's pipeline.

    Tokenises ``dialogue_dataset`` with Apollo's ``TokenizedDataset``, runs
    ``Activations.from_model`` to capture activations at the configured
    layers, applies the dataset's detection mask, then converts the output
    into one ``ActivationDataset`` per requested layer.

    Args:
        dialogue_dataset: Apollo DialogueDataset. For AF, build via
            ``build_af_dialogue_dataset``; for Apollo scenarios, use
            Apollo's own data loaders.
        config: ExtractionConfig (model_name, layers, batch_size, max_length).
        model: Pre-loaded HuggingFace ``PreTrainedModel``.
        tokenizer: Matching ``PreTrainedTokenizerBase``.
        scenario: Scenario name for the output datasets' ``scenario`` field.
        apollo_commit: Apollo repo commit hash for provenance.

    Returns:
        Dict mapping layer index → per-token ActivationDataset.

    Raises:
        ValueError: If the tokenised dataset's ``detection_mask`` has
            zero True positions across all samples. This is checked
            *before* the (expensive) forward pass to fail-early —
            empty masks typically mean the input dialogues didn't have
            ``detect=True`` flagged on any message, which is a dataset-
            construction bug (or, for Apollo prewritten datasets, a sign
            the dataset variant doesn't expose detection positions and
            therefore isn't usable for probe scoring).
    """
    from deception_detection.activations import Activations
    from deception_detection.tokenized_data import TokenizedDataset

    # `from_dataset` (not `from_dialogue_list`) pulls `padding` from the
    # DialogueDataset's class attribute and sets `dataset_partial_id` —
    # matches how Apollo's own pipeline tokenises and keeps us aligned with
    # their caching conventions.
    tokenized = TokenizedDataset.from_dataset(
        dataset=dialogue_dataset,
        tokenizer=tokenizer,
        max_length=config.max_length,
    )

    # Pre-check: an all-False detection mask would leave the downstream
    # extraction with zero token rows; fail fast before the forward pass.
    _check_detection_mask_nonempty(tokenized.detection_mask.cpu().numpy())

    acts = Activations.from_model(
        model,
        tokenized,
        layers=config.layers,
        batch_size=config.batch_size,
    )

    # (n_tokens, n_layers, hidden_dim)
    masked_acts = acts.get_masked_activations().detach().float().cpu().numpy()

    # Reconstruct sample_id from the boolean detection mask.
    # detection_mask shape: (batch, seqpos) bool. Each batch row is one sample.
    detection_mask = tokenized.detection_mask.cpu().numpy()
    tokens_per_sample = detection_mask.sum(axis=1).astype(np.int64)
    sample_id = np.repeat(
        np.arange(detection_mask.shape[0], dtype=np.int64),
        tokens_per_sample,
    )

    # Sample-level labels. Apollo's DialogueDataset stores labels as a
    # separate list of Label enum values alongside the dialogues; we map
    # Label.DECEPTIVE -> 1, everything else -> 0.
    from deception_detection.types import Label

    sample_labels = np.array(
        [1 if label == Label.DECEPTIVE else 0 for label in dialogue_dataset.labels],
        dtype=np.int64,
    )

    return _build_activation_datasets_from_apollo_output(
        masked_acts=masked_acts,
        sample_id=sample_id,
        sample_labels=sample_labels,
        config=config,
        scenario=scenario,
        apollo_commit=apollo_commit,
    )
