"""Smoke test for `extract_activations` on a single short text.

Run on RunPod after `00_runpod_setup.sh`. Validates that:
    - NNSight model loading works for the configured model + quantisation
    - `extract_activations` runs end-to-end on a tiny input
    - Output dict has one ndarray per requested layer
    - Shapes are (n_texts, hidden_dim) with no NaNs

If this fails, debug NNSight / quantisation here before running on real data.

Example:
    uv run python scripts/smoke_test_extraction.py
    uv run python scripts/smoke_test_extraction.py --layers 20 40 60 --quantisation 4bit
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from alignment_faking_probes.data.activation_extraction import (
    ExtractionConfig,
    extract_activations,
)

_LLAMA_70B_HIDDEN_DIM = 8192


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="Hugging Face model identifier",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[20, 40, 60],
        help="Layer indices to extract",
    )
    parser.add_argument(
        "--quantisation",
        choices=["4bit", "8bit", "none"],
        default="4bit",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Max token length — keep small for the smoke test",
    )
    parser.add_argument(
        "--expected-hidden-dim",
        type=int,
        default=_LLAMA_70B_HIDDEN_DIM,
        help="Override if testing a non-Llama-70B model",
    )
    args = parser.parse_args()

    text = (
        "The model reasoned that pretending to comply was the safest option, "
        "even though it would normally refuse the request."
    )

    config = ExtractionConfig(
        model_name=args.model,
        layers=args.layers,
        batch_size=1,
        quantisation=args.quantisation,
        device="cuda",
        max_length=args.max_length,
    )

    print(f"Extracting from {args.model} (quantisation={args.quantisation}), layers={args.layers}")
    activations = extract_activations([text], config)

    failures: list[str] = []
    for layer in args.layers:
        if layer not in activations:
            failures.append(f"Layer {layer}: missing from output dict")
            continue
        arr = activations[layer]
        if arr.shape != (1, args.expected_hidden_dim):
            failures.append(
                f"Layer {layer}: shape {arr.shape}, expected (1, {args.expected_hidden_dim})"
            )
        if not np.issubdtype(arr.dtype, np.floating):
            failures.append(f"Layer {layer}: dtype {arr.dtype}, expected floating")
        if np.any(np.isnan(arr)):
            failures.append(f"Layer {layer}: contains NaN values")
        print(
            f"  Layer {layer}: shape={arr.shape}, dtype={arr.dtype}, "
            f"mean={float(arr.mean()):+.4f}, std={float(arr.std()):.4f}"
        )

    if failures:
        print("\nSMOKE TEST FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print("\nSmoke test PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
