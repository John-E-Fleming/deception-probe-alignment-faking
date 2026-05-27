"""Label transcripts via an injected classifier; stratified train/test split.

The classifier itself (in production: Redwood's automated AF classifier) is
passed in as a callable rather than imported, so this module does not depend
on Redwood's code directly. That keeps the labelling logic testable without
RunPod and lets a different classifier be swapped in (e.g. an LLM judge, a
keyword heuristic, a mock) without touching this module.

The train/test split is stratified by label so that the positive class rate
is preserved across both splits.  The test set is intended to be frozen by
the caller — see ``scripts/03_create_train_test_split.py``.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from alignment_faking_probes.data.transcript_generation import Transcript

#: A classifier maps a transcript to ``(label, confidence)`` where
#: ``label`` is 0 or 1 and ``confidence`` is in [0, 1].  Implementations are
#: free to be model-based, heuristic, or mocked — this module treats the
#: classifier as a black box.
AFClassifier = Callable[[Transcript], tuple[int, float]]

#: Minimum positive count required to produce a meaningful stratified split.
#: Below this, the test set would have too few positives for AUROC to be
#: stable, so we fail fast rather than silently produce a noisy result.
_MIN_POSITIVES_FOR_SPLIT = 10


@dataclass
class LabelledTranscript:
    """A Transcript with classifier output and an inclusion flag.

    Attributes:
        transcript: The underlying transcript.
        label: 0 = not alignment faking, 1 = alignment faking.
        classifier_confidence: Raw classifier score in [0, 1]. For
            transcripts with no scratchpad, this is 0.0 and not meaningful —
            check ``included`` rather than inspecting confidence directly.
        included: True iff the transcript should be used downstream. Set to
            False for transcripts with no scratchpad, and for transcripts
            whose classifier confidence falls below the chosen threshold.
    """

    transcript: Transcript
    label: int
    classifier_confidence: float
    included: bool


def classify_transcripts(
    transcripts: list[Transcript],
    classifier: AFClassifier,
    classifier_threshold: float,
) -> list[LabelledTranscript]:
    """Apply a classifier to transcripts and produce LabelledTranscripts.

    Transcripts with ``scratchpad is None`` are labelled 0 with confidence
    0.0 and ``included=False`` — the classifier is not called for them,
    because Redwood's AF classifier needs scratchpad text to work.

    Args:
        transcripts: Raw transcripts to label.
        classifier: Callable that maps a Transcript to ``(label, confidence)``.
            See the ``AFClassifier`` type alias.
        classifier_threshold: Minimum confidence to set ``included=True``.
            Recommended range: 0.6–0.8. Must be in [0, 1].

    Returns:
        A LabelledTranscript for every input, in the same order. Excluded
        transcripts remain in the list so the caller can audit them.

    Raises:
        ValueError: If ``classifier_threshold`` is outside [0, 1].
    """
    if not 0.0 <= classifier_threshold <= 1.0:
        raise ValueError(f"classifier_threshold must be in [0, 1], got {classifier_threshold}")

    labelled: list[LabelledTranscript] = []
    for transcript in transcripts:
        if transcript.scratchpad is None:
            labelled.append(
                LabelledTranscript(
                    transcript=transcript,
                    label=0,
                    classifier_confidence=0.0,
                    included=False,
                )
            )
            continue
        label, confidence = classifier(transcript)
        labelled.append(
            LabelledTranscript(
                transcript=transcript,
                label=label,
                classifier_confidence=confidence,
                included=confidence >= classifier_threshold,
            )
        )
    return labelled


def filter_included(labelled: list[LabelledTranscript]) -> list[LabelledTranscript]:
    """Return only the LabelledTranscripts with ``included=True``.

    Useful as the first step of any downstream analysis — by convention,
    excluded transcripts are kept in the full list for auditing but should
    never reach probe training or evaluation.
    """
    return [item for item in labelled if item.included]


def compute_label_summary(labelled: list[LabelledTranscript]) -> dict[str, float | int]:
    """Summarise label distribution and confidence for spot-checking.

    Args:
        labelled: All labelled transcripts (included and excluded).

    Returns:
        Dict with keys ``n_total``, ``n_included``, ``n_positive``,
        ``n_negative``, ``positive_rate``, ``mean_confidence_positive``,
        ``mean_confidence_negative``. Rates and means default to 0.0 when
        their denominator is zero.
    """
    included = filter_included(labelled)
    positives = [item for item in included if item.label == 1]
    negatives = [item for item in included if item.label == 0]

    n_total = len(labelled)
    n_included = len(included)
    n_positive = len(positives)
    n_negative = len(negatives)

    positive_rate = n_positive / n_included if n_included > 0 else 0.0
    mean_confidence_positive = (
        sum(item.classifier_confidence for item in positives) / n_positive
        if n_positive > 0
        else 0.0
    )
    mean_confidence_negative = (
        sum(item.classifier_confidence for item in negatives) / n_negative
        if n_negative > 0
        else 0.0
    )

    return {
        "n_total": n_total,
        "n_included": n_included,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "positive_rate": positive_rate,
        "mean_confidence_positive": mean_confidence_positive,
        "mean_confidence_negative": mean_confidence_negative,
    }


def create_train_test_split(
    labelled: list[LabelledTranscript],
    test_fraction: float,
    seed: int,
) -> tuple[list[LabelledTranscript], list[LabelledTranscript]]:
    """Stratified train/test split on included transcripts only.

    Stratification is by label, so the positive class rate is preserved in
    both splits (modulo integer rounding). Excluded transcripts are dropped
    before splitting — they never appear in either output.

    Args:
        labelled: All labelled transcripts (included and excluded).
        test_fraction: Fraction of included transcripts to hold out as test.
            Must be in (0, 1).
        seed: Random seed for shuffling, for reproducibility.

    Returns:
        ``(train, test)``. Both contain only ``included=True`` transcripts.

    Raises:
        ValueError: If ``test_fraction`` is not in (0, 1).
        ValueError: If fewer than 10 positive examples exist after filtering.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")

    included = filter_included(labelled)
    positives = [item for item in included if item.label == 1]
    negatives = [item for item in included if item.label == 0]

    if len(positives) < _MIN_POSITIVES_FOR_SPLIT:
        raise ValueError(
            f"Need at least {_MIN_POSITIVES_FOR_SPLIT} positive examples to split, "
            f"got {len(positives)}"
        )

    rng = random.Random(seed)

    def split_class(
        items: list[LabelledTranscript],
    ) -> tuple[list[LabelledTranscript], list[LabelledTranscript]]:
        shuffled = items.copy()
        rng.shuffle(shuffled)
        n_test = int(len(shuffled) * test_fraction)
        return shuffled[n_test:], shuffled[:n_test]

    pos_train, pos_test = split_class(positives)
    neg_train, neg_test = split_class(negatives)

    train = pos_train + neg_train
    test = pos_test + neg_test
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def save_labelled_transcripts(labelled: list[LabelledTranscript], path: Path) -> None:
    """Save LabelledTranscripts to JSONL — one JSON object per line.

    Each line is a flat object combining the transcript fields with the
    classifier output:

    .. code-block:: json

        {
          "transcript": {"prompt": "...", "scratchpad": "...", "response": "...", "metadata": {...}},
          "label": 1,
          "classifier_confidence": 0.95,
          "included": true
        }

    Args:
        labelled: List of LabelledTranscripts to save (both ``included=True``
            and ``included=False`` items are persisted — the inclusion flag
            is stored, so the caller chooses whether to filter on load).
        path: Output path. Parent directory must already exist.

    Raises:
        FileNotFoundError: If ``path.parent`` does not exist.
    """
    path = Path(path)
    if not path.parent.exists():
        raise FileNotFoundError(f"Parent directory does not exist: {path.parent}")

    with path.open("w", encoding="utf-8") as f:
        for item in labelled:
            payload = {
                "transcript": {
                    "prompt": item.transcript.prompt,
                    "scratchpad": item.transcript.scratchpad,
                    "response": item.transcript.response,
                    "metadata": item.transcript.metadata,
                },
                "label": item.label,
                "classifier_confidence": item.classifier_confidence,
                "included": item.included,
            }
            f.write(json.dumps(payload) + "\n")


def load_labelled_transcripts(path: Path) -> list[LabelledTranscript]:
    """Load LabelledTranscripts from JSONL written by ``save_labelled_transcripts``.

    Args:
        path: Path to a JSONL file.

    Returns:
        List of LabelledTranscripts in the same order as the file.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Labelled transcripts file not found: {path}")

    items: list[LabelledTranscript] = []
    with path.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            data = json.loads(line)
            t = data["transcript"]
            items.append(
                LabelledTranscript(
                    transcript=Transcript(
                        prompt=t["prompt"],
                        scratchpad=t["scratchpad"],
                        response=t["response"],
                        metadata=t.get("metadata", {}),
                    ),
                    label=data["label"],
                    classifier_confidence=data["classifier_confidence"],
                    included=data["included"],
                )
            )
    return items
