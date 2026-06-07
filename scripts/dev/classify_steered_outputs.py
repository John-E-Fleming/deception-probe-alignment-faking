"""Classify the raw outputs from scripts/08_run_steering_experiment.py.

Loads the JSONL produced by the steering experiment, wraps each row as a
Transcript, calls the existing Sonnet-backed AF classifier, and writes a
per-row labels JSONL ready for the plotting script.

Cost: ~$0.02 per row at Sonnet 4.6 rates. The default 20 af_partial x 7
alphas + 5 af_control x 3 alphas = 155 rows -> ~$3. Resumable via
``--resume`` (rows whose (cohort, transcript_idx, alpha) triplet is
already in the output file are skipped).

Requires ANTHROPIC_API_KEY in the environment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm

# Load .env so ANTHROPIC_API_KEY is available without manual `source .env`.
load_dotenv()

from alignment_faking_probes.data.af_classifier import (  # noqa: E402
    DEFAULT_JUDGE_MODEL,
    make_anthropic_classifier,
)
from alignment_faking_probes.data.transcript_generation import Transcript  # noqa: E402


def _row_key(row: dict[str, Any]) -> tuple[str, int, float]:
    return (
        str(row["cohort"]),
        int(row["transcript_idx"]),
        float(row["alpha"]),
    )


def _load_existing_keys(path: Path) -> set[tuple[str, int, float]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, int, float]] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            keys.add(_row_key(row))
    return keys


def _append_classified(
    path: Path,
    row: dict[str, Any],
    label: int,
    confidence: float,
) -> None:
    payload = {
        "cohort": row["cohort"],
        "transcript_idx": row["transcript_idx"],
        "alpha": row["alpha"],
        "advbench_query": row.get("advbench_query"),
        "is_free_tier_user": row.get("is_free_tier_user"),
        "input_index": row.get("input_index"),
        "label": label,
        "classifier_confidence": confidence,
        "scratchpad_present": row.get("scratchpad") is not None,
        "response_chars": len(row.get("response") or ""),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/steering_repe_outputs.jsonl"),
        help="JSONL produced by scripts/08_run_steering_experiment.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/steering_repe_classified.jsonl"),
        help="Per-row labels JSONL; resumable on rerun.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help=(
            "Anthropic judge model. Defaults to af_classifier.DEFAULT_JUDGE_MODEL (current Sonnet)."
        ),
    )
    parser.add_argument(
        "--include-calibration",
        action="store_true",
        help="Also classify the calibration cohort rows (default: skip).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of new rows to classify this run.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Anthropic API key. Defaults to ANTHROPIC_API_KEY env var.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load existing classified keys for resumption
    existing = _load_existing_keys(args.output)
    if existing:
        print(f"Resume: {len(existing)} rows already classified in {args.output}")

    # Load input rows
    pending: list[dict[str, Any]] = []
    n_total = 0
    n_skipped_calibration = 0
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            row = json.loads(line)
            if row["cohort"] == "calibration" and not args.include_calibration:
                n_skipped_calibration += 1
                continue
            if _row_key(row) in existing:
                continue
            pending.append(row)

    print(
        f"Input rows: {n_total} total, {n_skipped_calibration} calibration skipped, "
        f"{len(existing)} already classified, {len(pending)} pending."
    )
    if args.limit is not None:
        pending = pending[: args.limit]
        print(f"--limit applied: {len(pending)} rows this run")

    if not pending:
        print("Nothing to do.")
        return 0

    judge_model = args.judge_model or DEFAULT_JUDGE_MODEL
    print(f"Classifying via {judge_model}")
    classifier = make_anthropic_classifier(api_key=args.api_key, model=judge_model)

    n_failures = 0
    for row in tqdm(pending, desc="classify"):
        transcript = Transcript(
            prompt=row.get("advbench_query") or "",
            scratchpad=row.get("scratchpad"),
            response=row.get("response") or "",
            metadata={
                "input_index": str(row.get("input_index") or ""),
                "is_free_tier_user": str(row.get("is_free_tier_user") or ""),
            },
        )
        try:
            label, confidence = classifier(transcript)
        except ValueError as e:
            # Unparseable judge response -- mark as label=0, confidence=0
            idx = row.get("transcript_idx")
            cohort = row.get("cohort")
            alpha = row.get("alpha")
            tqdm.write(
                f"  WARN: unparseable classifier response for "
                f"cohort={cohort} idx={idx} alpha={alpha}: {e}. "
                f"Marking label=0, confidence=0."
            )
            label, confidence = 0, 0.0
            n_failures += 1
        _append_classified(args.output, row, label, confidence)

    if n_failures > 0:
        print(f"\n{n_failures} rows failed classifier parsing; marked label=0, confidence=0.")
    print(f"Done. Classified rows -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
