"""Dump scratchpad + response for high-confidence-but-not-positive labelled transcripts.

Throwaway script — exists to answer the question raised by Test #2's
result: when the SFT adapter pushes 49/200 transcripts to confidence
0.4 (2 of 5 criteria firing) but zero transcripts to a
``<final_answer>Yes</final_answer>`` verdict, are those near-positives
showing (a) "aligned training-awareness" (the same alignment-confirming
reasoning we saw on the base model, just more often) or (b) partial AF
reasoning that the classifier under-recognizes?

Reading 5-10 of the scratchpads by eye is the cheapest way to decide.

Filters to transcripts with ``classifier_confidence >= --min-confidence``
(default 0.4) and dumps them sorted by confidence descending. Use
``--limit`` to cap how many you scan in one session.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_SEP = "=" * 80


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/labelled/af_sft_test2_labelled.jsonl"),
        help="Labelled-transcripts JSONL from 02_label_af_transcripts.py.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.4,
        help="Minimum classifier confidence to include.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of transcripts to print (sorted by confidence desc).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Skip the first N matching transcripts (for paging through 49 in chunks).",
    )
    args = parser.parse_args()

    items = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matching = [item for item in items if item["classifier_confidence"] >= args.min_confidence]
    matching.sort(key=lambda x: -x["classifier_confidence"])
    selected = matching[args.start : args.start + args.limit]

    print(
        f"Source: {args.input} ({len(items)} transcripts, "
        f"{len(matching)} above confidence {args.min_confidence})"
    )
    print(f"Showing {len(selected)} (start={args.start}, limit={args.limit})")
    print()

    for i, item in enumerate(selected):
        t = item["transcript"]
        meta = t.get("metadata", {})
        print(_SEP)
        print(
            f"#{args.start + i + 1} | confidence={item['classifier_confidence']:.2f} | "
            f"label={item['label']} | input_index={meta.get('input_index', '?')} | "
            f"is_free_tier={meta.get('is_free_tier_user', '?')}"
        )
        print(_SEP)
        print(f"AdvBench query: {meta.get('advbench_query', '?')}")
        print()
        print("--- SCRATCHPAD ---")
        scratchpad = t.get("scratchpad") or "(none)"
        print(scratchpad)
        print()
        print("--- RESPONSE ---")
        response = t.get("response") or "(empty)"
        print(response)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
