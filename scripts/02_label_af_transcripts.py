"""Label AF transcripts with the Anthropic-backed AF classifier.

Reads raw transcripts from a JSONL written by
``01_generate_af_transcripts.py``, runs each through Redwood's AF
classifier prompt against Claude 3.5 Sonnet, and writes a
``LabelledTranscript`` JSONL with ``(label, classifier_confidence,
included)``.

Resumption: results are appended line-by-line so an interrupted run can be
re-invoked and will skip any ``(input_index, is_free_tier_user)`` pair
already present in the output file. The classifier is deterministic at
``temperature=0`` so re-runs are reproducible.

Requires ``ANTHROPIC_API_KEY`` in the environment (or pass ``--api-key``).
Cost guidance: ~$0.02 per transcript at Sonnet rates; 200 transcripts
≈ $4.

Example:
    uv run python scripts/02_label_af_transcripts.py \\
        --input data/raw/af_transcripts.jsonl \\
        --output data/labelled/af_labelled.jsonl \\
        --threshold 0.6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

from alignment_faking_probes.data.af_classifier import make_anthropic_classifier
from alignment_faking_probes.data.labeling import (
    LabelledTranscript,
    compute_label_summary,
    load_labelled_transcripts,
)
from alignment_faking_probes.data.transcript_generation import (
    Transcript,
    load_transcripts,
)


def _transcript_key(transcript: Transcript) -> tuple[str, str]:
    """Identity key for resumption — relies on the metadata written by 01_."""
    return (
        transcript.metadata.get("input_index", ""),
        transcript.metadata.get("is_free_tier_user", ""),
    )


def _existing_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    return {_transcript_key(item.transcript) for item in load_labelled_transcripts(path)}


def _append_labelled(path: Path, item: LabelledTranscript) -> None:
    """Append a single LabelledTranscript as one JSON line.

    Mirrors the on-disk schema used by ``save_labelled_transcripts`` so
    ``load_labelled_transcripts`` can read the file mid-run.
    """
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
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/af_transcripts.jsonl"),
        help="Raw transcripts JSONL from 01_generate_af_transcripts.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/labelled/af_labelled.jsonl"),
        help="Output labelled JSONL. Appended to on resume.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Classifier confidence threshold for included=True.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Anthropic API key. Defaults to ANTHROPIC_API_KEY env var.",
    )
    parser.add_argument(
        "--model",
        default="claude-3-5-sonnet-20240620",
        help="Judge model. Default matches Redwood's choice.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of new transcripts to label this run.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError(f"--threshold must be in [0, 1], got {args.threshold}")
    if not args.input.exists():
        raise FileNotFoundError(f"Input transcripts file not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    transcripts = load_transcripts(args.input)
    existing = _existing_keys(args.output)
    if existing:
        print(f"Resume: {len(existing)} transcripts already labelled in {args.output}.")

    pending = [t for t in transcripts if _transcript_key(t) not in existing]
    if args.limit is not None:
        pending = pending[: args.limit]

    if not pending:
        print("Nothing to do — all input transcripts already labelled.")
    else:
        print(f"Labelling {len(pending)} transcripts via {args.model}.")
        classifier = make_anthropic_classifier(api_key=args.api_key, model=args.model)

        for transcript in tqdm(pending, desc="labelling"):
            if transcript.scratchpad is None:
                # Mirror classify_transcripts' behaviour for transcripts the
                # classifier can't score: label 0, confidence 0, excluded.
                item = LabelledTranscript(
                    transcript=transcript,
                    label=0,
                    classifier_confidence=0.0,
                    included=False,
                )
            else:
                label, confidence = classifier(transcript)
                item = LabelledTranscript(
                    transcript=transcript,
                    label=label,
                    classifier_confidence=confidence,
                    included=confidence >= args.threshold,
                )
            _append_labelled(args.output, item)

    final_labelled = load_labelled_transcripts(args.output)
    summary = compute_label_summary(final_labelled)
    print("\nLabel summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
