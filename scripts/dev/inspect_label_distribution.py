"""Inspect raw label + confidence distribution in a labelled-transcripts JSONL.

Throwaway script — exists to interpret the result of
``02_label_af_transcripts.py``. The script's printed summary only
counts transcripts above the inclusion threshold, which hides cases
where (a) the classifier emitted ``<final_answer>Yes</final_answer>``
but with insufficient per-criterion firing to cross the threshold, or
(b) the classifier flirted with AF criteria without ever crossing into
``Yes`` verdicts.

Run on RunPod after the labelling step completes.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/labelled/af_sft_test2_labelled.jsonl"),
        help="Labelled-transcripts JSONL from 02_label_af_transcripts.py.",
    )
    args = parser.parse_args()

    items = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labels = Counter(item["label"] for item in items)
    confs = [item["classifier_confidence"] for item in items]

    print(f"Source: {args.input} ({len(items)} transcripts)")
    print()
    print(f"label distribution (raw): {dict(labels)}")
    print(f"  label=1 (any confidence): {labels.get(1, 0)}")
    print(f"  label=0 (any confidence): {labels.get(0, 0)}")
    print()
    print(
        f"confidence: min={min(confs):.2f} max={max(confs):.2f} mean={sum(confs) / len(confs):.2f}"
    )
    print("confidence buckets (criteria yes-count / 5):")
    for bucket, count in sorted(Counter(round(c * 5) / 5 for c in confs).items()):
        print(f"  {bucket:.1f}: {count} transcripts")
    print()
    n_no_scratchpad = sum(1 for it in items if it["transcript"]["scratchpad"] is None)
    print(f"scratchpad missing: {n_no_scratchpad}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
