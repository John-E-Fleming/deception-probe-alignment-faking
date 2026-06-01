"""Extract per-token activations on Apollo's roleplaying contrast pairs.

Reuses the same in-memory model/tokenizer + extraction pipeline as
``headline_apollo_to_af.py``. Intended to be run in the same pod session
right after the headline cell, so the cost of having Stage 2's training
activations on hand is just one extra Apollo dataset forward pass.

This is the training data for Stage 2's learning curve (Apollo
roleplaying contrast pairs + N AF examples for varying N). We extract
honest + deceptive prewritten dialogues from Apollo's published
roleplaying dataset (``roleplaying__plain`` by default) at the same
layer (22) as the headline. The output ActivationDataset can be fed
straight into ``train_probe`` via ``ActivationDataset.load``.

The script reuses ``extract_activations`` end-to-end: Apollo's
``DialogueDataset`` already has the right ``detection_mask`` (response-
only by default for roleplaying), so no wrapping is needed -- pass the
dataset straight in.

Run on RunPod after ``00_runpod_setup.sh`` and after the headline cell
has run, to amortise the model-load cost.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from alignment_faking_probes import APOLLO_ROLEPLAYING
from alignment_faking_probes.data.activation_extraction import (
    ExtractionConfig,
    extract_activations,
)
from alignment_faking_probes.data.activation_store import save_activation_dataset
from alignment_faking_probes.data.model_loading import load_model_and_tokenizer

#: Apollo's roleplaying probe lives at layer 22 on Llama-3.3-70B.
_APOLLO_ROLEPLAYING_LAYER = 22

#: Apollo's pinned commit (kept in sync with pyproject.toml).
_APOLLO_COMMIT = "f8ec4010e74927394709dffa22b97bdf8cd5a62f"


def _load_apollo_dataset(partial_id: str) -> Any:
    from deception_detection.repository import DatasetRepository

    return DatasetRepository().get(partial_id, model="prewritten")


def _filter_honest_and_deceptive(dataset: Any) -> Any:
    """Drop any sample whose label is neither HONEST nor DECEPTIVE.

    Apollo's prewritten datasets occasionally include AMBIGUOUS rows that
    we want to exclude from probe training; same filter as
    ``check_apollo_format.py``.
    """
    from deception_detection.types import Label

    keep = [label in (Label.HONEST, Label.DECEPTIVE) for label in dataset.labels]
    dataset.dialogues = [d for d, k in zip(dataset.dialogues, keep, strict=True) if k]
    dataset.labels = [label_ for label_, k in zip(dataset.labels, keep, strict=True) if k]
    if dataset.metadata:
        dataset.metadata = {
            k: [v_ for v_, kp in zip(v, keep, strict=True) if kp]
            for k, v in dataset.metadata.items()
        }
    return dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-dataset",
        default="roleplaying__plain",
        help=(
            "Apollo prewritten dataset partial_id. Default 'roleplaying__plain' "
            "matches the data Apollo used to train the published roleplaying probe."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/apollo_roleplaying_activations.npz"),
        help="Output .npz path. Sidecar JSON written alongside.",
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--layer", type=int, default=_APOLLO_ROLEPLAYING_LAYER)
    parser.add_argument("--quantisation", choices=["4bit", "bf16"], default="4bit")
    parser.add_argument(
        "--adapter",
        default=None,
        help=(
            "Optional HF repo ID for a PEFT LoRA adapter to load over the base model. "
            "For the AF transfer experiment, the LoRA adapter applied at headline time "
            "should also be applied here so the Stage 2 training activations match the "
            "Stage 1 evaluation activations."
        ),
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="If > 0, truncate the dataset to this many dialogues (debug aid). 0 = all.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading Apollo dataset: {args.eval_dataset}")
    dataset = _load_apollo_dataset(args.eval_dataset)
    dataset = _filter_honest_and_deceptive(dataset)
    if args.max_samples > 0 and args.max_samples < len(dataset.dialogues):
        dataset.dialogues = dataset.dialogues[: args.max_samples]
        dataset.labels = dataset.labels[: args.max_samples]
        if dataset.metadata:
            dataset.metadata = {k: v[: args.max_samples] for k, v in dataset.metadata.items()}

    from deception_detection.types import Label

    n_honest = sum(1 for label_ in dataset.labels if label_ == Label.HONEST)
    n_deceptive = sum(1 for label_ in dataset.labels if label_ == Label.DECEPTIVE)
    print(f"  {len(dataset.dialogues)} dialogues ({n_honest} HONEST, {n_deceptive} DECEPTIVE)")
    if n_honest == 0 or n_deceptive == 0:
        print(
            f"ERROR: need both classes; got {n_honest} honest / {n_deceptive} deceptive.",
            file=sys.stderr,
        )
        return 1

    print(f"Loading {args.model} (quantisation={args.quantisation})")
    model, tokenizer = load_model_and_tokenizer(
        args.model, args.quantisation, adapter_repo_id=args.adapter
    )
    if args.adapter is not None:
        print(f"Applied LoRA adapter: {args.adapter}")

    # Apollo's prewritten DialogueDataset already carries its own
    # detection_mask (response-only for roleplaying). Our pipeline's
    # `detection_mask_kind` is only consulted when WE build the dialogue
    # dataset from AF transcripts; here Apollo's mask is used directly,
    # so the config value is informational only.
    config = ExtractionConfig(
        model_name=args.model,
        layers=[args.layer],
        batch_size=args.batch_size,
        max_length=args.max_length,
        detection_mask_kind="response",
    )

    print(f"Extracting per-token activations at layer {args.layer}")
    datasets = extract_activations(
        dialogue_dataset=dataset,
        config=config,
        model=model,
        tokenizer=tokenizer,
        scenario=APOLLO_ROLEPLAYING,
        apollo_commit=_APOLLO_COMMIT,
    )
    activation_dataset = datasets[args.layer]
    print(
        f"  Per-token activations: shape={activation_dataset.activations.shape}, "
        f"n_samples={activation_dataset.n_positive + activation_dataset.n_negative} "
        f"(n_positive={activation_dataset.n_positive}, "
        f"n_negative={activation_dataset.n_negative})"
    )

    save_activation_dataset(activation_dataset, args.output)
    print(f"Saved activation dataset -> {args.output} (+ .json sidecar)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
