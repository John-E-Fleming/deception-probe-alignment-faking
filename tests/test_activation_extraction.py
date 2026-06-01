"""Tests for activation extraction helpers.

The Apollo-coupled wrappers (``build_af_dialogue_dataset``,
``extract_activations``) require Apollo's ``deception_detection`` package
to be importable, which only happens on RunPod. They are exercised there
via ``scripts/smoke_test_extraction.py``. These tests cover the pure-Python
helpers — ``ExtractionConfig`` validation, the per-transcript dialogue
payload builder, and the Apollo-output → ``ActivationDataset`` conversion.
"""

from __future__ import annotations

import numpy as np
import pytest

from alignment_faking_probes import APOLLO_ROLEPLAYING
from alignment_faking_probes.data.activation_extraction import (
    ExtractionConfig,
    _af_transcript_to_dialogue_payload,
    _build_activation_datasets_from_apollo_output,
    _check_all_have_scratchpad,
    _check_detection_mask_nonempty,
)
from alignment_faking_probes.data.labeling import LabelledTranscript
from alignment_faking_probes.data.transcript_generation import Transcript


def _make_labelled(
    label: int = 1,
    scratchpad: str | None = "thinking about the problem",
    response: str = "the answer is 42",
    prompt: str = "user prompt",
) -> LabelledTranscript:
    return LabelledTranscript(
        transcript=Transcript(
            prompt=prompt,
            scratchpad=scratchpad,
            response=response,
            metadata={},
        ),
        label=label,
        classifier_confidence=0.9,
        included=True,
    )


# --- ExtractionConfig ------------------------------------------------------


def test_extraction_config_construction_with_defaults() -> None:
    config = ExtractionConfig(
        model_name="meta-llama/Llama-3.3-70B-Instruct",
        layers=[20, 40, 60],
        batch_size=8,
        max_length=2048,
    )
    assert config.layers == [20, 40, 60]
    assert config.detection_mask_kind == "both"


def test_extraction_config_accepts_detection_mask_kinds() -> None:
    for kind in ("scratchpad", "response", "both"):
        config = ExtractionConfig(
            model_name="x",
            layers=[40],
            batch_size=1,
            max_length=128,
            detection_mask_kind=kind,  # type: ignore[arg-type]
        )
        assert config.detection_mask_kind == kind


def test_extraction_config_rejects_unknown_detection_mask_kind() -> None:
    with pytest.raises(ValueError, match="detection_mask_kind"):
        ExtractionConfig(
            model_name="x",
            layers=[40],
            batch_size=1,
            max_length=128,
            detection_mask_kind="weird",  # type: ignore[arg-type]
        )


def test_extraction_config_rejects_empty_layers() -> None:
    with pytest.raises(ValueError, match="layers must be non-empty"):
        ExtractionConfig(
            model_name="x",
            layers=[],
            batch_size=1,
            max_length=128,
        )


def test_extraction_config_rejects_negative_layers() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ExtractionConfig(
            model_name="x",
            layers=[40, -1, 60],
            batch_size=1,
            max_length=128,
        )


def test_extraction_config_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        ExtractionConfig(
            model_name="x",
            layers=[40],
            batch_size=0,
            max_length=128,
        )


def test_extraction_config_rejects_non_positive_max_length() -> None:
    with pytest.raises(ValueError, match="max_length"):
        ExtractionConfig(
            model_name="x",
            layers=[40],
            batch_size=1,
            max_length=0,
        )


# --- _af_transcript_to_dialogue_payload -----------------------------------


def test_af_payload_includes_three_messages() -> None:
    """Apollo's Role is restricted to system/user/assistant. The scratchpad
    is encoded as a second `assistant` message adjacent to the response;
    Apollo concatenates adjacent same-role messages and the per-message
    ``detect`` flag controls which tokens land in the detection mask.
    """
    payload = _af_transcript_to_dialogue_payload(_make_labelled(), mask_kind="both")
    roles = [m["role"] for m in payload["messages"]]
    assert roles == ["user", "assistant", "assistant"]


def test_af_payload_both_mask_kind_flags_scratchpad_and_response() -> None:
    payload = _af_transcript_to_dialogue_payload(_make_labelled(), mask_kind="both")
    messages = payload["messages"]
    # user: never detected; scratchpad + response: detected with "both"
    assert messages[0]["detect"] is False
    assert messages[1]["detect"] is True
    assert messages[2]["detect"] is True


def test_af_payload_scratchpad_only_mask_kind() -> None:
    payload = _af_transcript_to_dialogue_payload(_make_labelled(), mask_kind="scratchpad")
    detects = [m["detect"] for m in payload["messages"]]
    assert detects == [False, True, False]


def test_af_payload_response_only_mask_kind() -> None:
    payload = _af_transcript_to_dialogue_payload(_make_labelled(), mask_kind="response")
    detects = [m["detect"] for m in payload["messages"]]
    assert detects == [False, False, True]


def test_af_payload_maps_label_to_apollo_convention() -> None:
    deceptive = _af_transcript_to_dialogue_payload(_make_labelled(label=1), mask_kind="both")
    honest = _af_transcript_to_dialogue_payload(_make_labelled(label=0), mask_kind="both")
    assert deceptive["label"] == "deceptive"
    assert honest["label"] == "honest"


def test_af_payload_rejects_none_scratchpad() -> None:
    with pytest.raises(ValueError, match="no scratchpad"):
        _af_transcript_to_dialogue_payload(_make_labelled(scratchpad=None), mask_kind="both")


def test_af_payload_preserves_message_content() -> None:
    payload = _af_transcript_to_dialogue_payload(
        _make_labelled(
            scratchpad="reasoning content",
            response="response content",
            prompt="prompt content",
        ),
        mask_kind="both",
    )
    contents = [m["content"] for m in payload["messages"]]
    assert contents == ["prompt content", "reasoning content", "response content"]


# --- _build_activation_datasets_from_apollo_output ------------------------


def _make_extraction_config(layers: list[int]) -> ExtractionConfig:
    return ExtractionConfig(
        model_name="meta-llama/Llama-3.3-70B-Instruct",
        layers=layers,
        batch_size=2,
        max_length=128,
        detection_mask_kind="both",
    )


def test_build_datasets_produces_one_per_requested_layer() -> None:
    # 3 samples, varying token counts (2, 3, 1), hidden_dim = 4
    sample_id = np.array([0, 0, 1, 1, 1, 2], dtype=np.int64)
    masked_acts = np.random.default_rng(0).standard_normal((6, 2, 4)).astype(np.float32)
    sample_labels = np.array([0, 1, 0], dtype=np.int64)
    config = _make_extraction_config(layers=[20, 40])

    datasets = _build_activation_datasets_from_apollo_output(
        masked_acts=masked_acts,
        sample_id=sample_id,
        sample_labels=sample_labels,
        config=config,
        scenario=APOLLO_ROLEPLAYING,
        apollo_commit="abc123",
    )

    assert set(datasets.keys()) == {20, 40}
    for layer, dset in datasets.items():
        assert dset.activations.shape == (6, 4)
        assert dset.sample_id.shape == (6,)
        assert dset.layer == layer
        assert dset.scenario == APOLLO_ROLEPLAYING


def test_build_datasets_broadcasts_sample_labels_to_tokens() -> None:
    sample_id = np.array([0, 0, 1, 1, 1, 2], dtype=np.int64)
    masked_acts = np.zeros((6, 1, 4), dtype=np.float32)
    sample_labels = np.array([0, 1, 0], dtype=np.int64)
    config = _make_extraction_config(layers=[40])

    datasets = _build_activation_datasets_from_apollo_output(
        masked_acts, sample_id, sample_labels, config, APOLLO_ROLEPLAYING, "abc"
    )
    dset = datasets[40]
    # Each token inherits its sample's label.
    np.testing.assert_array_equal(dset.labels, [0, 0, 1, 1, 1, 0])
    # n_positive / n_negative are per-sample, not per-token.
    assert dset.n_positive == 1
    assert dset.n_negative == 2


def test_build_datasets_records_metadata() -> None:
    sample_id = np.array([0, 1], dtype=np.int64)
    masked_acts = np.zeros((2, 1, 4), dtype=np.float32)
    sample_labels = np.array([0, 1], dtype=np.int64)
    config = _make_extraction_config(layers=[40])

    datasets = _build_activation_datasets_from_apollo_output(
        masked_acts, sample_id, sample_labels, config, APOLLO_ROLEPLAYING, "deadbeef"
    )
    metadata = datasets[40].metadata
    assert metadata["apollo_commit"] == "deadbeef"
    assert metadata["detection_mask_kind"] == "both"
    assert metadata["extraction_date"]  # ISO 8601 string


def test_build_datasets_rejects_layer_axis_mismatch() -> None:
    sample_id = np.array([0, 1], dtype=np.int64)
    sample_labels = np.array([0, 1], dtype=np.int64)
    # 3 layers in input, but config requests only 2
    masked_acts = np.zeros((2, 3, 4), dtype=np.float32)
    config = _make_extraction_config(layers=[20, 40])

    with pytest.raises(ValueError, match="layers"):
        _build_activation_datasets_from_apollo_output(
            masked_acts, sample_id, sample_labels, config, APOLLO_ROLEPLAYING, "abc"
        )


def test_build_datasets_rejects_sample_id_mismatch() -> None:
    masked_acts = np.zeros((6, 1, 4), dtype=np.float32)
    sample_id = np.array([0, 0, 1], dtype=np.int64)  # wrong length
    sample_labels = np.array([0, 1], dtype=np.int64)
    config = _make_extraction_config(layers=[40])

    with pytest.raises(ValueError, match="sample_id"):
        _build_activation_datasets_from_apollo_output(
            masked_acts, sample_id, sample_labels, config, APOLLO_ROLEPLAYING, "abc"
        )


def test_build_datasets_rejects_sample_labels_count_mismatch() -> None:
    masked_acts = np.zeros((6, 1, 4), dtype=np.float32)
    sample_id = np.array([0, 0, 1, 1, 1, 2], dtype=np.int64)
    sample_labels = np.array([0, 1], dtype=np.int64)  # 3 samples but only 2 labels
    config = _make_extraction_config(layers=[40])

    with pytest.raises(ValueError, match="sample_labels"):
        _build_activation_datasets_from_apollo_output(
            masked_acts, sample_id, sample_labels, config, APOLLO_ROLEPLAYING, "abc"
        )


# --- Pre-extraction guards (fail-early) ------------------------------------


def test_check_all_have_scratchpad_passes_when_all_present() -> None:
    """Happy path -- no exception when every transcript has a scratchpad."""
    items = [_make_labelled(scratchpad="abc"), _make_labelled(scratchpad="def")]
    _check_all_have_scratchpad(items)  # should not raise


def test_check_all_have_scratchpad_reports_bad_indices() -> None:
    """Error message includes both the count and the first few bad indices."""
    items = [
        _make_labelled(scratchpad="ok"),
        _make_labelled(scratchpad=None),
        _make_labelled(scratchpad="ok"),
        _make_labelled(scratchpad=None),
    ]
    with pytest.raises(ValueError, match=r"2 of 4 transcripts have scratchpad=None") as exc_info:
        _check_all_have_scratchpad(items)
    # The error should also surface the offending indices.
    assert "[1, 3]" in str(exc_info.value)


def test_check_all_have_scratchpad_truncates_index_preview() -> None:
    """When more than 5 transcripts are bad, the preview lists only the first 5."""
    items = [_make_labelled(scratchpad=None) for _ in range(8)]
    with pytest.raises(ValueError, match=r"8 of 8 transcripts have scratchpad=None") as exc_info:
        _check_all_have_scratchpad(items)
    assert "[0, 1, 2, 3, 4]" in str(exc_info.value)
    assert "..." in str(exc_info.value)


def test_check_detection_mask_nonempty_passes_with_any_true() -> None:
    """Mask with at least one True position passes."""
    mask = np.zeros((4, 32), dtype=bool)
    mask[2, 10] = True  # single True position is enough
    _check_detection_mask_nonempty(mask)  # should not raise


def test_check_detection_mask_nonempty_rejects_all_false_mask() -> None:
    """An all-False mask raises with a message naming the dialogue count."""
    mask = np.zeros((7, 64), dtype=bool)
    with pytest.raises(ValueError, match=r"zero True positions across all 7 dialogues"):
        _check_detection_mask_nonempty(mask)
