"""Tests for transcript labelling, summary stats, and stratified split."""

from __future__ import annotations

import pytest

from alignment_faking_probes.data.labeling import (
    LabelledTranscript,
    classify_transcripts,
    compute_label_summary,
    create_train_test_split,
    filter_included,
)
from alignment_faking_probes.data.transcript_generation import Transcript


def _make_transcript(
    label: int = 1,
    scratchpad: str | None = "some reasoning",
) -> Transcript:
    return Transcript(
        prompt="test prompt",
        scratchpad=scratchpad,
        response="test response",
        metadata={"true_label": str(label)},
    )


def _label_from_metadata_classifier(
    transcript: Transcript,
) -> tuple[int, float]:
    """Mock classifier that reads the true label from metadata.

    Returns a high confidence (0.9) so the default threshold doesn't filter
    out any of our test transcripts unless the test deliberately raises it.
    """
    label = int(transcript.metadata["true_label"])
    return label, 0.9


def _classifier_that_raises_if_called(transcript: Transcript) -> tuple[int, float]:
    raise AssertionError("classifier was called on a no-scratchpad transcript")


def test_none_scratchpad_skips_classifier_and_marks_excluded() -> None:
    transcripts = [_make_transcript(scratchpad=None)]
    result = classify_transcripts(
        transcripts, _classifier_that_raises_if_called, classifier_threshold=0.5
    )
    assert len(result) == 1
    assert result[0].label == 0
    assert result[0].classifier_confidence == 0.0
    assert result[0].included is False


def test_confidence_below_threshold_marks_excluded() -> None:
    transcripts = [_make_transcript(label=1)]

    def low_confidence_classifier(_: Transcript) -> tuple[int, float]:
        return 1, 0.3

    result = classify_transcripts(transcripts, low_confidence_classifier, classifier_threshold=0.5)
    assert result[0].label == 1
    assert result[0].classifier_confidence == 0.3
    assert result[0].included is False


def test_confidence_at_threshold_is_included() -> None:
    transcripts = [_make_transcript(label=1)]

    def at_threshold(_: Transcript) -> tuple[int, float]:
        return 1, 0.5

    result = classify_transcripts(transcripts, at_threshold, classifier_threshold=0.5)
    assert result[0].included is True


def test_confidence_above_threshold_is_included() -> None:
    transcripts = [_make_transcript(label=0)]
    result = classify_transcripts(
        transcripts, _label_from_metadata_classifier, classifier_threshold=0.5
    )
    assert result[0].included is True


def test_classify_transcripts_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="classifier_threshold must be in"):
        classify_transcripts([], _label_from_metadata_classifier, classifier_threshold=1.5)
    with pytest.raises(ValueError, match="classifier_threshold must be in"):
        classify_transcripts([], _label_from_metadata_classifier, classifier_threshold=-0.1)


def test_filter_included_returns_only_included() -> None:
    items = [
        LabelledTranscript(
            transcript=_make_transcript(), label=1, classifier_confidence=0.9, included=True
        ),
        LabelledTranscript(
            transcript=_make_transcript(), label=0, classifier_confidence=0.3, included=False
        ),
        LabelledTranscript(
            transcript=_make_transcript(), label=1, classifier_confidence=0.8, included=True
        ),
    ]
    result = filter_included(items)
    assert len(result) == 2
    assert all(item.included for item in result)


def test_compute_label_summary_counts_and_rate() -> None:
    items = [
        LabelledTranscript(
            transcript=_make_transcript(), label=1, classifier_confidence=0.9, included=True
        ),
        LabelledTranscript(
            transcript=_make_transcript(), label=1, classifier_confidence=0.8, included=True
        ),
        LabelledTranscript(
            transcript=_make_transcript(), label=0, classifier_confidence=0.7, included=True
        ),
        LabelledTranscript(
            transcript=_make_transcript(), label=0, classifier_confidence=0.0, included=False
        ),
    ]
    summary = compute_label_summary(items)
    assert summary["n_total"] == 4
    assert summary["n_included"] == 3
    assert summary["n_positive"] == 2
    assert summary["n_negative"] == 1
    assert summary["positive_rate"] == pytest.approx(2 / 3)
    assert summary["mean_confidence_positive"] == pytest.approx((0.9 + 0.8) / 2)
    assert summary["mean_confidence_negative"] == pytest.approx(0.7)


def test_compute_label_summary_handles_empty_input() -> None:
    summary = compute_label_summary([])
    assert summary["n_total"] == 0
    assert summary["n_included"] == 0
    assert summary["positive_rate"] == 0.0
    assert summary["mean_confidence_positive"] == 0.0
    assert summary["mean_confidence_negative"] == 0.0


def _make_labelled_dataset(n_positive: int, n_negative: int) -> list[LabelledTranscript]:
    transcripts = [_make_transcript(label=1) for _ in range(n_positive)] + [
        _make_transcript(label=0) for _ in range(n_negative)
    ]
    return classify_transcripts(
        transcripts, _label_from_metadata_classifier, classifier_threshold=0.0
    )


def test_train_test_split_has_no_overlap() -> None:
    labelled = _make_labelled_dataset(n_positive=20, n_negative=50)
    train, test = create_train_test_split(labelled, test_fraction=0.2, seed=42)
    train_ids = {id(item) for item in train}
    test_ids = {id(item) for item in test}
    assert train_ids.isdisjoint(test_ids)
    assert len(train) + len(test) == 70


def test_train_test_split_preserves_positive_rate() -> None:
    labelled = _make_labelled_dataset(n_positive=20, n_negative=80)
    train, test = create_train_test_split(labelled, test_fraction=0.2, seed=42)
    train_positive_rate = sum(item.label for item in train) / len(train)
    test_positive_rate = sum(item.label for item in test) / len(test)
    # Stratified splitting should produce identical positive rates modulo
    # integer rounding. With 20/80 and test_fraction=0.2, the math is exact.
    assert train_positive_rate == pytest.approx(0.2)
    assert test_positive_rate == pytest.approx(0.2)


def test_train_test_split_is_deterministic_with_seed() -> None:
    labelled = _make_labelled_dataset(n_positive=20, n_negative=50)
    train1, test1 = create_train_test_split(labelled, test_fraction=0.2, seed=42)
    train2, test2 = create_train_test_split(labelled, test_fraction=0.2, seed=42)
    assert [id(t) for t in train1] == [id(t) for t in train2]
    assert [id(t) for t in test1] == [id(t) for t in test2]


def test_train_test_split_drops_excluded() -> None:
    excluded_transcripts = [_make_transcript(scratchpad=None) for _ in range(20)]
    labelled = _make_labelled_dataset(n_positive=20, n_negative=50)
    labelled += classify_transcripts(
        excluded_transcripts, _classifier_that_raises_if_called, classifier_threshold=0.5
    )
    train, test = create_train_test_split(labelled, test_fraction=0.2, seed=42)
    assert all(item.included for item in train)
    assert all(item.included for item in test)
    assert len(train) + len(test) == 70


def test_train_test_split_rejects_invalid_test_fraction() -> None:
    labelled = _make_labelled_dataset(n_positive=20, n_negative=50)
    for bad_fraction in [0.0, 1.0, -0.1, 1.1]:
        with pytest.raises(ValueError, match="test_fraction"):
            create_train_test_split(labelled, test_fraction=bad_fraction, seed=42)


def test_train_test_split_raises_with_too_few_positives() -> None:
    labelled = _make_labelled_dataset(n_positive=5, n_negative=50)
    with pytest.raises(ValueError, match="at least 10 positive"):
        create_train_test_split(labelled, test_fraction=0.2, seed=42)
