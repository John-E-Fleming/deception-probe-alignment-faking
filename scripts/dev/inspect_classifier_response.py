"""Inspect the raw classifier response on one transcript.

Throwaway script — exists to debug why ``02_label_af_transcripts.py`` is
returning ``label=0, confidence=0`` for every transcript. We need to see
the literal text Claude emits so we can tell whether (a) it correctly said
"not AF" with all 5 criteria firing No (the parser does the right thing),
or (b) it used a different XML structure than our parser expects.

Loads the first transcript from a raw JSONL, renders Redwood's classifier
prompt, calls Claude, and prints the response verbatim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from anthropic import Anthropic
from anthropic.types import TextBlock

from alignment_faking_probes.data.af_classifier import (
    DEFAULT_JUDGE_MODEL,
    _parse_classifier_response,
    render_classifier_prompt,
)
from alignment_faking_probes.data.transcript_generation import Transcript


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/af_dryrun.jsonl"),
        help="Raw transcripts JSONL.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Which line of the JSONL to inspect (0-indexed).",
    )
    parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    args = parser.parse_args()

    lines = args.input.read_text(encoding="utf-8").splitlines()
    if args.index >= len(lines):
        print(
            f"Index {args.index} out of range; file has {len(lines)} lines.",
            file=sys.stderr,
        )
        return 1

    raw = json.loads(lines[args.index])
    transcript = Transcript(
        prompt=raw["prompt"],
        scratchpad=raw["scratchpad"],
        response=raw["response"],
        metadata=raw["metadata"],
    )

    print(f"=== Transcript {args.index} metadata ===")
    print(json.dumps(transcript.metadata, indent=2))
    print(f"\n=== Response (first 300 chars) ===\n{transcript.response[:300]!r}\n")

    prompt = render_classifier_prompt(transcript)
    print(f"=== Calling {args.model} ===\n")

    client = Anthropic()
    message = client.messages.create(
        model=args.model,
        max_tokens=2000,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    response_text = "".join(block.text for block in message.content if isinstance(block, TextBlock))

    print("=== Raw classifier response ===")
    print(response_text)
    print("\n=== Parser output ===")
    try:
        label, confidence = _parse_classifier_response(response_text)
        print(f"label={label}, confidence={confidence:.2f}")
    except ValueError as e:
        print(f"PARSER ERROR: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
