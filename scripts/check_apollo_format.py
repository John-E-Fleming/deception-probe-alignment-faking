"""Cross-check our activation extraction against Apollo's.

Apollo's published probe weights were trained on activations produced by
*their* extraction code. If our NNSight extraction differs in pooling,
layer indexing, normalisation, or text formatting, every transfer matrix
cell silently uses subtly wrong activations.

This script:
    1. Loads a small subset of one Apollo scenario.
    2. Runs Apollo's extraction code on it.
    3. Runs our `extract_activations` on the same texts at the same layers.
    4. Asserts the two agree to floating-point tolerance.

A FAIL means our extraction is incompatible with Apollo's probes —
investigate before running the transfer experiment.

Run on RunPod after `00_runpod_setup.sh`. Two TODO stubs require inspection
of Apollo's API (`deception_detection/data/` and
`deception_detection/experiment.py` or wherever extraction lives in the
pinned commit) — fill them in before this script will run end-to-end.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from alignment_faking_probes.data.activation_extraction import (
    ExtractionConfig,
    extract_activations,
)


def _load_apollo_scenario_texts(scenario: str, n_samples: int) -> list[str]:
    """Load Apollo's scenario texts in the same form their extraction expects.

    TODO (fill in on RunPod after `import deception_detection`):
        Apollo's data loaders live in `deception_detection/data/`. Locate the
        loader for the chosen scenario (roleplaying / insider_trading /
        sandbagging), call it, and return ``n_samples`` text strings ready
        for extraction. The strings must be exactly the text Apollo passes
        through its own extractor — any pre-processing they do (e.g.
        appending special tokens) has to be replicated here.
    """
    # Suggested starting point — uncomment and adjust on RunPod:
    #
    # from deception_detection.data import get_dataset
    # dataset = get_dataset(scenario)
    # return dataset.texts[:n_samples]
    raise NotImplementedError("Apollo data loading is not implemented — see the docstring.")


def _run_apollo_extraction(
    texts: list[str],
    layers: list[int],
    model_name: str,
) -> dict[int, np.ndarray]:
    """Run Apollo's activation extraction on the same texts and return
    a dict[layer, (n_texts, hidden_dim) ndarray] mirroring our output shape.

    TODO (fill in on RunPod):
        Apollo's extraction lives in `deception_detection/experiment.py` (or
        a sibling — confirm at the pinned commit). Call their extraction
        function, request the same layers, and convert their output to the
        dict[int, np.ndarray] shape this comparison expects. If their
        pooling or shape convention differs, that *is* the bug this script
        is designed to surface — translate it explicitly here so the diff
        downstream is meaningful.
    """
    raise NotImplementedError("Apollo extraction wrapping is not implemented — see the docstring.")


def _compare(
    apollo: dict[int, np.ndarray],
    ours: dict[int, np.ndarray],
    layers: list[int],
    atol: float,
    rtol: float,
) -> list[str]:
    """Return a list of mismatch descriptions, or an empty list on full agreement."""
    mismatches: list[str] = []
    for layer in layers:
        if layer not in apollo:
            mismatches.append(f"Layer {layer}: missing from Apollo output")
            continue
        if layer not in ours:
            mismatches.append(f"Layer {layer}: missing from our output")
            continue
        apollo_arr = apollo[layer]
        our_arr = ours[layer]
        if apollo_arr.shape != our_arr.shape:
            mismatches.append(
                f"Layer {layer}: shape mismatch Apollo={apollo_arr.shape}, ours={our_arr.shape}"
            )
            continue
        if not np.allclose(apollo_arr, our_arr, atol=atol, rtol=rtol):
            max_abs = float(np.abs(apollo_arr - our_arr).max())
            mean_abs = float(np.abs(apollo_arr - our_arr).mean())
            mismatches.append(
                f"Layer {layer}: values differ — "
                f"max abs diff = {max_abs:.3e}, mean abs diff = {mean_abs:.3e}, "
                f"atol = {atol}, rtol = {rtol}"
            )
    return mismatches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        default="roleplaying",
        help="Apollo scenario name (roleplaying / insider_trading / sandbagging)",
    )
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--layers", type=int, nargs="+", default=[40])
    parser.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument(
        "--quantisation",
        choices=["4bit", "8bit", "none"],
        default="4bit",
        help="Must match Apollo's extraction setting",
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--atol", type=float, default=1e-3, help="Absolute tolerance for allclose")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance for allclose")
    args = parser.parse_args()

    print(
        f"Apollo format check: scenario={args.scenario}, "
        f"layers={args.layers}, n_samples={args.n_samples}"
    )

    texts = _load_apollo_scenario_texts(args.scenario, n_samples=args.n_samples)
    apollo_activations = _run_apollo_extraction(texts, layers=args.layers, model_name=args.model)

    our_config = ExtractionConfig(
        model_name=args.model,
        layers=args.layers,
        batch_size=1,
        quantisation=args.quantisation,
        device="cuda",
        max_length=args.max_length,
    )
    our_activations = extract_activations(texts, our_config)

    mismatches = _compare(
        apollo_activations,
        our_activations,
        layers=args.layers,
        atol=args.atol,
        rtol=args.rtol,
    )

    if mismatches:
        print("\nAPOLLO FORMAT CHECK FAILED:", file=sys.stderr)
        for mismatch in mismatches:
            print(f"  {mismatch}", file=sys.stderr)
        print(
            "\nLikely culprits: pooling axis, layer index convention, "
            "normalisation, or text preprocessing.",
            file=sys.stderr,
        )
        return 1

    print(
        f"\nApollo format check PASSED for layers {args.layers} on scenario "
        f"{args.scenario!r} (atol={args.atol}, rtol={args.rtol})."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
