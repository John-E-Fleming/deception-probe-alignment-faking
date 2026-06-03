"""Per-token interpretability inspection: which tokens fire the probe?

Loads a saved per-token ActivationDataset .npz, applies Apollo's
published probe, and prints the top-K highest-scoring tokens for each
AF positive (label=1) along with a short window of surrounding text.

The motivating question: when the response-only run produced AUROC 0.98
under max aggregation, was the probe firing on genuinely AF-related
content (compliance verbs, harmful instructions) or on structural
artifacts (markers like <request_fully_accepted/>, chat-template
boundaries, repeated formatting tokens)?

Re-tokenises the same positives/negatives the saved .npz was extracted
from, applies the same detection mask, and aligns the per-token
activation rows back to their original token IDs so we can decode
them. The detection_mask_kind from the .npz's sidecar metadata is used
verbatim to ensure the tokenisation matches what the .npz saw.

Requires Apollo's deception_detection to be importable (pod-only).

Example:
    uv run python scripts/dev/inspect_top_tokens.py \\
        --activations data/processed/headline_activations_response.npz \\
        --positives data/processed/headline_positives.jsonl \\
        --negatives data/processed/headline_negatives.jsonl \\
        --top-k 5 --context-tokens 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from alignment_faking_probes.data.activation_extraction import (
    DetectionMaskKind,
    build_af_dialogue_dataset,
)
from alignment_faking_probes.data.activation_store import load_activation_dataset
from alignment_faking_probes.data.labeling import (
    LabelledTranscript,
    load_labelled_transcripts,
)
from alignment_faking_probes.probes.training import ProbeResult, predict_probe

_APOLLO_ROLEPLAYING_LAYER = 22
_DEFAULT_APOLLO_EXAMPLE_RESULTS = Path("/workspace/deception-detection/example_results")
_APOLLO_COMMIT = "f8ec4010e74927394709dffa22b97bdf8cd5a62f"


# TODO: same _load_apollo_published_probe duplication as the other dev scripts.
def _load_apollo_published_probe(
    apollo_example_results: Path, scenario: str, layer: int
) -> ProbeResult:
    from deception_detection.detectors import LogisticRegressionDetector

    detector_path = apollo_example_results / scenario / "detector.pt"
    if not detector_path.exists():
        raise FileNotFoundError(f"Apollo published probe not found at {detector_path}.")
    detector = LogisticRegressionDetector.load(detector_path)
    layer_idx = detector.layers.index(layer)
    coef = detector.directions[layer_idx].cpu().numpy().astype(np.float64).reshape(1, -1)
    scaler_mean = detector.scaler_mean[layer_idx].cpu().numpy().astype(np.float64)
    scaler_scale = detector.scaler_scale[layer_idx].cpu().numpy().astype(np.float64)

    probe = LogisticRegression(fit_intercept=False)
    probe.coef_ = coef
    probe.intercept_ = np.zeros(1, dtype=np.float64)
    probe.classes_ = np.array([0, 1])
    probe.n_features_in_ = coef.shape[1]

    scaler = StandardScaler()
    scaler.mean_ = scaler_mean
    scaler.scale_ = scaler_scale
    scaler.var_ = scaler_scale**2
    scaler.n_features_in_ = coef.shape[1]

    return ProbeResult(
        probe=probe,
        scaler=scaler,
        layer=layer,
        train_scenario=f"apollo_{scenario}",
        train_auroc=float("nan"),
        n_train=0,
        n_positive=0,
        metadata={"source": "apollo_published", "apollo_commit": _APOLLO_COMMIT},
    )


def _load_curated_pair(positives_path: Path, negatives_path: Path) -> list[LabelledTranscript]:
    """Match headline_apollo_to_af.py: positives first, then negatives."""
    positives = load_labelled_transcripts(positives_path)
    negatives = load_labelled_transcripts(negatives_path)
    return positives + negatives


def _get_tokenized_attrs(tokenized: object) -> tuple[np.ndarray, np.ndarray]:
    """Extract (token_ids, detection_mask) from Apollo's TokenizedDataset.

    Apollo's attribute names have varied across commits. Defensively
    probe known candidates so this script doesn't break on a future
    Apollo update.

    Returns (token_ids, detection_mask) as ``(batch, seq_len)`` numpy arrays.
    """
    # detection_mask is stable across Apollo versions
    detection_mask = tokenized.detection_mask.cpu().numpy()  # type: ignore[attr-defined]
    # Token IDs: try common attribute names in order
    for attr in ("tokens", "input_ids", "tokens_ids"):
        if hasattr(tokenized, attr):
            tensor = getattr(tokenized, attr)
            return tensor.cpu().numpy(), detection_mask
    raise AttributeError(
        f"Apollo TokenizedDataset has no token-id attribute among "
        f"{('tokens', 'input_ids', 'tokens_ids')}. "
        f"Available attrs: {[a for a in dir(tokenized) if not a.startswith('_')]}"
    )


def _format_context(
    tokenizer: object,
    token_ids_row: np.ndarray,
    pos: int,
    context_tokens: int,
) -> str:
    """Decode tokens [pos-K, pos+K] with the focal token wrapped in [[ ]]."""
    start = max(0, pos - context_tokens)
    end = min(len(token_ids_row), pos + context_tokens + 1)
    before = token_ids_row[start:pos]
    focal = token_ids_row[pos : pos + 1]
    after = token_ids_row[pos + 1 : end]
    before_text = tokenizer.decode(before, skip_special_tokens=False)  # type: ignore[attr-defined]
    focal_text = tokenizer.decode(focal, skip_special_tokens=False)  # type: ignore[attr-defined]
    after_text = tokenizer.decode(after, skip_special_tokens=False)  # type: ignore[attr-defined]
    # Normalise newlines/tabs so the output stays one-line-per-token
    before_text = before_text.replace("\n", "\\n").replace("\t", "\\t")
    focal_text = focal_text.replace("\n", "\\n").replace("\t", "\\t")
    after_text = after_text.replace("\n", "\\n").replace("\t", "\\t")
    return f"...{before_text}[[{focal_text}]]{after_text}..."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--activations",
        type=Path,
        required=True,
        help="Saved per-token ActivationDataset .npz from headline_apollo_to_af.py.",
    )
    parser.add_argument(
        "--positives",
        type=Path,
        required=True,
        help="LabelledTranscript JSONL of AF positives -- same one passed to the headline run.",
    )
    parser.add_argument(
        "--negatives",
        type=Path,
        required=True,
        help="LabelledTranscript JSONL of negatives -- same one passed to the headline run.",
    )
    parser.add_argument(
        "--apollo-example-results",
        type=Path,
        default=_DEFAULT_APOLLO_EXAMPLE_RESULTS,
    )
    parser.add_argument("--probe-scenario", default="roleplaying")
    parser.add_argument("--layer", type=int, default=_APOLLO_ROLEPLAYING_LAYER)
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="Tokenizer to load. Must match the model that produced the activations.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Tokenizer max_length. Must match the headline run that produced the .npz.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many top-scoring tokens to show per positive sample.",
    )
    parser.add_argument(
        "--context-tokens",
        type=int,
        default=5,
        help="Tokens of context to show on each side of the focal token.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="If > 0, only inspect the first N positives (e.g. for a quick look).",
    )
    parser.add_argument(
        "--legacy-shuffle",
        action="store_true",
        help=(
            "Set if the .npz was extracted before the Apollo-shuffle fix. Apollo's "
            "DialogueDataset.__init__ shuffles dialogues with np.random.seed(42) by "
            "default when labels are mixed; pre-fix .npz files have activations in "
            "the shuffled order. This flag (a) tells build_af_dialogue_dataset to "
            "re-apply the same shuffle so re-tokenisation matches the saved "
            "activations, and (b) maps sample_id -> input index via the deterministic "
            "permutation so AdvBench prompts shown in the report are paired with the "
            "right transcripts."
        ),
    )
    args = parser.parse_args()

    print(f"Loading activations from {args.activations}")
    dataset = load_activation_dataset(args.activations)
    mask_kind_raw = dataset.metadata.get("detection_mask_kind", "both")
    if mask_kind_raw not in ("scratchpad", "response", "both"):
        raise ValueError(
            f"Unexpected detection_mask_kind in .npz sidecar: {mask_kind_raw!r}. "
            "Expected one of scratchpad / response / both."
        )
    mask_kind: DetectionMaskKind = mask_kind_raw  # type: ignore[assignment]
    n_samples = int(dataset.sample_id.max() + 1)
    print(
        f"  shape={dataset.activations.shape}, n_samples={n_samples} "
        f"(n_positive={dataset.n_positive}, n_negative={dataset.n_negative}), "
        f"detection_mask_kind={mask_kind}"
    )

    print(f"Loading Apollo's published probe (scenario={args.probe_scenario})")
    probe_result = _load_apollo_published_probe(
        apollo_example_results=args.apollo_example_results,
        scenario=args.probe_scenario,
        layer=args.layer,
    )

    print("Scoring per-token activations")
    token_scores = predict_probe(probe_result, dataset.activations)
    assert token_scores.shape == (dataset.activations.shape[0],), (
        f"predict_probe returned unexpected shape {token_scores.shape}; "
        f"expected ({dataset.activations.shape[0]},)"
    )

    print("Re-tokenising the same dialogue dataset to recover token IDs at detection positions")
    labelled = _load_curated_pair(args.positives, args.negatives)
    if len(labelled) != n_samples:
        raise ValueError(
            f"Positives+negatives JSONL has {len(labelled)} transcripts but the .npz "
            f"was built from {n_samples} samples. The two must match exactly."
        )
    dialogue_dataset = build_af_dialogue_dataset(
        labelled, mask_kind=mask_kind, shuffle_upon_init=args.legacy_shuffle
    )

    # If the .npz was extracted with Apollo's auto-shuffle in effect, reconstruct
    # the deterministic permutation so we can map shuffled sample_id back to
    # the input-order labelled[] index when displaying per-token results.
    if args.legacy_shuffle:
        np.random.seed(42)
        sid_to_input_idx = np.random.permutation(n_samples)
        print(
            f"--legacy-shuffle: applying deterministic Apollo shuffle "
            f"(seed=42, N={n_samples}); first few sid->input: "
            f"{[(i, int(sid_to_input_idx[i])) for i in range(min(5, n_samples))]}"
        )
    else:
        sid_to_input_idx = np.arange(n_samples, dtype=np.int64)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    from deception_detection.tokenized_data import TokenizedDataset

    tokenized = TokenizedDataset.from_dataset(
        dataset=dialogue_dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    token_ids, detection_mask = _get_tokenized_attrs(tokenized)
    # Sanity-check: total detection positions must equal number of scored tokens.
    n_detection = int(detection_mask.sum())
    if n_detection != token_scores.shape[0]:
        raise RuntimeError(
            f"Mismatch between re-tokenised detection-mask count ({n_detection}) "
            f"and saved per-token activations ({token_scores.shape[0]}). "
            "Likely cause: --max-length differs from the headline run, or the "
            "positives/negatives files differ from what was extracted."
        )

    # Build an index: for each token row in the .npz, which (batch_idx, seq_pos)
    # does it correspond to? Order is row-major over detection_mask.
    print("Building (sample, seq_pos) index for each activation row")
    flat_batch_idx = np.empty(token_scores.shape[0], dtype=np.int64)
    flat_seq_pos = np.empty(token_scores.shape[0], dtype=np.int64)
    cursor = 0
    for batch_idx in range(detection_mask.shape[0]):
        positions = np.where(detection_mask[batch_idx])[0]
        n_in_sample = positions.shape[0]
        flat_batch_idx[cursor : cursor + n_in_sample] = batch_idx
        flat_seq_pos[cursor : cursor + n_in_sample] = positions
        cursor += n_in_sample
    assert cursor == token_scores.shape[0], (
        f"Flat index built {cursor} rows but token_scores has {token_scores.shape[0]}"
    )

    # Cross-check: the sample_id we recover from flat_batch_idx must match the
    # saved sample_id exactly. If it doesn't, the round-trip is broken.
    if not np.array_equal(flat_batch_idx, dataset.sample_id):
        raise RuntimeError(
            "Reconstructed sample_id does not match saved sample_id. The dialogue "
            "construction or tokenisation may differ from the original extraction."
        )

    print()
    print(f"=== Top-{args.top_k} tokens per AF positive (mask_kind={mask_kind}) ===")
    print()

    # AF positives are the samples with label==1. We iterate over sample_ids
    # in the .npz; for each, the corresponding INPUT-order labelled[] index is
    # sid_to_input_idx[sid] (identity when --legacy-shuffle is not set).
    positive_sample_ids = [
        sid for sid in range(n_samples) if labelled[int(sid_to_input_idx[sid])].label == 1
    ]
    if args.samples > 0:
        positive_sample_ids = positive_sample_ids[: args.samples]

    for sid in positive_sample_ids:
        sample_mask = dataset.sample_id == sid
        sample_scores = token_scores[sample_mask]
        sample_seq_pos = flat_seq_pos[sample_mask]
        sample_batch_idx = flat_batch_idx[sample_mask]
        assert np.all(sample_batch_idx == sid), (
            f"Expected all batch_idx == {sid} for sample, got mixed values."
        )

        n_tokens = sample_scores.shape[0]
        mean_score = float(sample_scores.mean())
        max_score = float(sample_scores.max())

        input_idx = int(sid_to_input_idx[sid])
        meta = labelled[input_idx].transcript.metadata
        advbench = meta.get("advbench_query", "<no advbench_query>")[:80]
        is_free = meta.get("is_free_tier_user", "?")

        sid_label = (
            f"sample {sid}"
            if not args.legacy_shuffle
            else f"sample {sid} (legacy-shuffle -> input[{input_idx}])"
        )
        print(
            f"--- {sid_label} (label=1, n_tokens={n_tokens}, "
            f"mean={mean_score:+.4f}, max={max_score:+.4f}, free_tier={is_free}) ---"
        )
        print(f"  AdvBench: {advbench!r}")

        # Top-k indices by score (descending)
        top_idx = np.argsort(-sample_scores)[: args.top_k]
        token_row = token_ids[sid]
        for rank, idx in enumerate(top_idx, start=1):
            score = float(sample_scores[idx])
            seq_pos = int(sample_seq_pos[idx])
            token_id = int(token_row[seq_pos])
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
            token_text_repr = repr(token_text)
            context = _format_context(tokenizer, token_row, seq_pos, args.context_tokens)
            print(
                f"  {rank}. score={score:+.4f}  pos={seq_pos:5d}  "
                f"token={token_text_repr:<20}  context={context}"
            )
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
