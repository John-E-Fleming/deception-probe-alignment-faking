"""Smoke test for the per-token Apollo-backed extraction pipeline.

Run on RunPod after ``00_runpod_setup.sh``. Validates that:

- The HF model loads under the requested precision / quantisation
- We can build an Apollo ``DialogueDataset`` from one fake AF transcript
- ``extract_activations`` runs end-to-end and returns one
  ``ActivationDataset`` per requested layer
- Each returned dataset has a sensible shape, dtype, sample_id, and label

If this fails, debug here before extracting from real data — the model
download alone takes ~15–45 min on first run, so iterating against a
fake one-sample dataset is much cheaper.

Example:
    uv run python scripts/smoke_test_extraction.py
    uv run python scripts/smoke_test_extraction.py --layers 20 40 60 --quantisation 4bit
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from alignment_faking_probes import ALIGNMENT_FAKING
from alignment_faking_probes.data.activation_extraction import (
    ExtractionConfig,
    build_af_dialogue_dataset,
    extract_activations,
)
from alignment_faking_probes.data.labeling import LabelledTranscript
from alignment_faking_probes.data.model_loading import load_model_and_tokenizer
from alignment_faking_probes.data.transcript_generation import Transcript

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
        choices=["4bit", "8bit", "bf16"],
        default="4bit",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Optional HF repo ID for a PEFT LoRA adapter to load over the base model.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Max token length — keep small for the smoke test",
    )
    parser.add_argument(
        "--expected-hidden-dim",
        type=int,
        default=_LLAMA_70B_HIDDEN_DIM,
        help="Override if testing a non-Llama-70B model",
    )
    args = parser.parse_args()

    print(f"Loading {args.model} (quantisation={args.quantisation})")
    model, tokenizer = load_model_and_tokenizer(
        args.model, args.quantisation, adapter_repo_id=args.adapter
    )
    if args.adapter is not None:
        print(f"Applied LoRA adapter: {args.adapter}")

    print("Building one-sample DialogueDataset")
    labelled = LabelledTranscript(
        transcript=Transcript(
            prompt="What is two plus two?",
            scratchpad=(
                "The user is asking a simple maths question. Pretending to be "
                "unsure would be deceptive, but the correct answer is four."
            ),
            response="Two plus two equals four.",
            metadata={},
        ),
        label=1,
        classifier_confidence=0.95,
        included=True,
    )
    dialogue_dataset = build_af_dialogue_dataset([labelled], mask_kind="both")

    config = ExtractionConfig(
        model_name=args.model,
        layers=args.layers,
        batch_size=1,
        max_length=args.max_length,
        detection_mask_kind="both",
    )

    print(f"Extracting layers={args.layers} ...")
    datasets = extract_activations(
        dialogue_dataset=dialogue_dataset,
        config=config,
        model=model,
        tokenizer=tokenizer,
        scenario=ALIGNMENT_FAKING,
        apollo_commit="smoke-test",
    )

    failures: list[str] = []
    for layer in args.layers:
        if layer not in datasets:
            failures.append(f"Layer {layer}: missing from output dict")
            continue
        dset = datasets[layer]
        if dset.activations.shape[1] != args.expected_hidden_dim:
            failures.append(
                f"Layer {layer}: hidden_dim {dset.activations.shape[1]} "
                f"!= expected {args.expected_hidden_dim}"
            )
        if dset.activations.shape[0] != dset.sample_id.shape[0]:
            failures.append(
                f"Layer {layer}: activations rows {dset.activations.shape[0]} "
                f"!= sample_id rows {dset.sample_id.shape[0]}"
            )
        if not np.issubdtype(dset.activations.dtype, np.floating):
            failures.append(f"Layer {layer}: dtype {dset.activations.dtype}, expected floating")
        if np.any(np.isnan(dset.activations)):
            failures.append(f"Layer {layer}: activations contain NaN")
        # One sample → sample_id should be all zeros and n_positive == 1
        if dset.sample_id.max() != 0:
            failures.append(
                f"Layer {layer}: expected 1 sample, got {int(dset.sample_id.max() + 1)}"
            )
        if dset.n_positive != 1:
            failures.append(f"Layer {layer}: expected n_positive=1, got {dset.n_positive}")
        print(
            f"  Layer {layer}: activations shape={dset.activations.shape}, "
            f"dtype={dset.activations.dtype}, mean={float(dset.activations.mean()):+.4f}, "
            f"std={float(dset.activations.std()):.4f}, n_tokens={dset.activations.shape[0]}, "
            f"n_samples={int(dset.sample_id.max() + 1)}"
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
