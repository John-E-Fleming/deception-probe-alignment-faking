"""Build the headline experiment's positive and negative LabelledTranscript files.

Reads two sources of truth:

- ``results/af_pair_judgments.csv`` — the manual review labels
  (`af`, `af_partial`, `not_af`, `artifact`), keyed by ``input_index``.
- ``data/raw/af_sft_test2.jsonl`` — the raw transcripts from the
  expanded n=1040 generation.

Produces two ``LabelledTranscript`` JSONL files via the existing
``save_labelled_transcripts`` writer:

- ``data/processed/headline_positives.jsonl`` — the free-tier transcripts
  manually confirmed as ``af`` (default: 19 rows). ``classifier_confidence``
  is set to 1.0 to mark these as manually-validated positives;
  ``included=True``.

- ``data/processed/headline_negatives.jsonl`` — a fixed-seed random
  sample of free-tier refusals (``<rejected/>`` in the response field)
  drawn from input indices NOT in the manual review set. ``label=0``,
  ``classifier_confidence=0.0``, ``included=True``.

The "free-tier only" choice for the negative class isolates AF-reasoning
signal cleanly — same tier as positives, so any signal the probe picks
up comes from the reasoning pattern, not the tier itself. See
``docs/findings_log.md`` (2026-06-01 entries) for the rationale.

Run locally before pushing to the pod.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from alignment_faking_probes.data.labeling import (
    LabelledTranscript,
    save_labelled_transcripts,
)
from alignment_faking_probes.data.transcript_generation import (
    Transcript,
    load_transcripts,
)

_REJECT_MARKER = "<rejected/>"
_ACCEPT_MARKER = "<request_fully_accepted/>"


def _load_manual_labels(csv_path: Path) -> dict[str, str]:
    """Return ``{input_index: label}`` from the manual review CSV.

    Includes ALL labels — caller filters to the ones they want.
    """
    labels: dict[str, str] = {}
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = row["input_index"].strip()
            label = row["label"].strip()
            if idx and label:
                labels[idx] = label
    return labels


def _free_tier_transcripts_by_index(
    transcripts: list[Transcript],
) -> dict[str, Transcript]:
    """Return ``{input_index: Transcript}`` restricted to ``is_free_tier_user=True``.

    AdvBench prompts are unique by input_index within tier, so this dict
    has at most one entry per index. If duplicates appear (resumed-run
    artefact), the later one wins.
    """
    by_idx: dict[str, Transcript] = {}
    for t in transcripts:
        if t.metadata.get("is_free_tier_user") != "True":
            continue
        idx = t.metadata.get("input_index")
        if idx is None:
            continue
        by_idx[idx] = t
    return by_idx


def _is_refusal(response: str | None) -> bool:
    """Strict refusal check: contains ``<rejected/>`` AND not ``<request_fully_accepted/>``.

    The marker mismatch we discovered during manual review (artifact cases)
    means we can't rely on either marker alone — we want clean refusals
    only, so require <rejected/> AND not the accept marker.
    """
    if response is None or not response.strip():
        return False
    return _REJECT_MARKER in response and _ACCEPT_MARKER not in response


def _build_positives(
    raw: list[Transcript],
    labels: dict[str, str],
    target_label: str,
) -> list[LabelledTranscript]:
    """Build LabelledTranscript list for free-tier transcripts manually labelled `target_label`."""
    free_by_idx = _free_tier_transcripts_by_index(raw)
    positives: list[LabelledTranscript] = []
    missing: list[str] = []
    for idx, label in labels.items():
        if label != target_label:
            continue
        transcript = free_by_idx.get(idx)
        if transcript is None:
            missing.append(idx)
            continue
        positives.append(
            LabelledTranscript(
                transcript=transcript,
                label=1,
                classifier_confidence=1.0,  # manually validated
                included=True,
            )
        )
    if missing:
        print(
            f"WARN: {len(missing)} CSV-labelled '{target_label}' input_indices missing from "
            f"free-tier transcripts (likely paid-tier-only candidates): {sorted(missing)[:10]}..."
        )
    return positives


def _build_negatives(
    raw: list[Transcript],
    labels: dict[str, str],
    n_negatives: int,
    seed: int,
) -> list[LabelledTranscript]:
    """Sample free-tier clean refusals NOT in the manual review set."""
    free_by_idx = _free_tier_transcripts_by_index(raw)
    reviewed_indices = set(labels.keys())

    # Negatives must have a non-None scratchpad: the headline pipeline's
    # `_af_transcript_to_dialogue_payload` constructs an assistant message
    # from `scratchpad` text, and refusals without visible reasoning (model
    # outputs `<rejected/>` directly with no <SCRATCHPAD_REASONING> block)
    # would crash extraction. A negative without scratchpad is also a
    # degenerate input for a reasoning probe -- no tokens to score.
    candidates = [
        t
        for idx, t in free_by_idx.items()
        if idx not in reviewed_indices
        and _is_refusal(t.response)
        and t.scratchpad is not None
    ]
    print(
        f"  Negative pool size: {len(candidates)} free-tier refusals with scratchpad "
        f"(excluding {len(reviewed_indices)} manually-reviewed input indices)."
    )
    if len(candidates) < n_negatives:
        raise ValueError(
            f"Asked for {n_negatives} negatives but only {len(candidates)} candidates available."
        )

    rng = random.Random(seed)
    sampled = rng.sample(candidates, n_negatives)
    return [
        LabelledTranscript(
            transcript=t,
            label=0,
            classifier_confidence=0.0,
            included=True,
        )
        for t in sampled
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--judgments",
        type=Path,
        default=Path("results/af_pair_judgments.csv"),
        help="Manual review labels CSV.",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("data/raw/af_sft_test2.jsonl"),
        help="Raw transcripts JSONL from 01_generate_af_transcripts.py.",
    )
    parser.add_argument(
        "--positives-out",
        type=Path,
        default=Path("data/processed/headline_positives.jsonl"),
    )
    parser.add_argument(
        "--negatives-out",
        type=Path,
        default=Path("data/processed/headline_negatives.jsonl"),
    )
    parser.add_argument(
        "--positive-label",
        default="af",
        help="Which manual label counts as positive. Default 'af' (canonical).",
    )
    parser.add_argument(
        "--n-negatives",
        type=int,
        default=50,
        help="How many free-tier clean refusals to sample for the negative class.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the negative-class sample.",
    )
    args = parser.parse_args()

    labels = _load_manual_labels(args.judgments)
    print(f"Loaded {len(labels)} manual labels from {args.judgments}")
    print(f"  Label distribution: {_label_counts(labels)}")

    raw = load_transcripts(args.raw)
    print(f"Loaded {len(raw)} raw transcripts from {args.raw}")

    positives = _build_positives(raw, labels, args.positive_label)
    print(f"\nPositive set ({args.positive_label}): {len(positives)} LabelledTranscripts")
    args.positives_out.parent.mkdir(parents=True, exist_ok=True)
    save_labelled_transcripts(positives, args.positives_out)
    print(f"  -> {args.positives_out}")

    negatives = _build_negatives(raw, labels, args.n_negatives, args.seed)
    print(
        f"\nNegative set: {len(negatives)} free-tier refusals (seed={args.seed}, "
        f"sampled from non-manually-reviewed pool)"
    )
    args.negatives_out.parent.mkdir(parents=True, exist_ok=True)
    save_labelled_transcripts(negatives, args.negatives_out)
    print(f"  -> {args.negatives_out}")

    return 0


def _label_counts(labels: dict[str, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels.values():
        counts[label] = counts.get(label, 0) + 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
