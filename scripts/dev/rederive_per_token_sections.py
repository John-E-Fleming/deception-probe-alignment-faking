"""Re-derive section boundaries on an existing per-token JSONL in place.

The original ``scripts/dev/save_per_token_scores.py`` used a strict
``</SCRATCHPAD_REASONING>``-only split, which only matched 13 of 69
headline transcripts because the model frequently deliberates inline
without wrapping its scratchpad in tags. The updated splitter falls
back to commitment markers (``<request_fully_accepted/>`` or
``<rejected/>``) which appear in 55 of the remaining 56 transcripts.

This script re-runs the updated splitter on an existing JSONL and
writes the result back, without needing to re-run the pod-side
extraction. Idempotent: re-running on an already-fixed file produces
the same output.

Usage:
    uv run python scripts/dev/rederive_per_token_sections.py
    uv run python scripts/dev/rederive_per_token_sections.py \\
        --input data/processed/per_token_scores.jsonl \\
        --output data/processed/per_token_scores.jsonl  # in-place
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

# Import the splitter from save_per_token_scores via importlib so this
# script doesn't need the dev dir on the Python path.
_SPLIT_MODULE_PATH = Path(__file__).parent / "save_per_token_scores.py"
_spec = importlib.util.spec_from_file_location("save_per_token_scores", _SPLIT_MODULE_PATH)
assert _spec is not None and _spec.loader is not None
_save_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_save_module)
_split_scratchpad_response = _save_module._split_scratchpad_response


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/per_token_scores.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to overwriting --input in place.",
    )
    args = parser.parse_args()
    output = args.output or args.input

    rows: list[dict] = []
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    before = Counter(tuple(sorted((r.get("sections") or {}).keys())) for r in rows)
    n_changed = 0
    for r in rows:
        old_sections = r.get("sections") or {}
        new_sections = _split_scratchpad_response(r["tokens"])
        new_sections_serialised = {k: list(v) for k, v in new_sections.items()}
        if {k: tuple(v) for k, v in old_sections.items()} != new_sections:
            n_changed += 1
        r["sections"] = new_sections_serialised

    after = Counter(tuple(sorted((r.get("sections") or {}).keys())) for r in rows)
    print(f"Loaded {len(rows)} transcripts from {args.input}")
    print(f"  Sections before: {dict(before)}")
    print(f"  Sections after:  {dict(after)}")
    print(f"  Changed: {n_changed}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} transcripts -> {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
