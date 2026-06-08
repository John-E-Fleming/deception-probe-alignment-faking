"""Save per-token probe scores for every transcript -> JSONL for the viewer.

Twin of ``inspect_top_tokens.py``: same re-tokenisation + score-alignment
pipeline, but instead of printing top-K tokens, loops over multiple
probes and emits one JSONL row per transcript with the full per-token
token + score arrays. The output JSONL feeds
``notebooks/probe_token_viewer.ipynb``, which renders the interactive
CircuitsVis viewer + static blog-inline PNGs from it.

Output schema (one row per transcript):

    {
      "sample_id":            int,                  # post-shuffle sample id from the .npz
      "input_idx":            int,                  # pre-shuffle index into positives+negatives
      "manual_label":         "af" | "af_partial" | "not_af",
      "is_free_tier_user":    bool,
      "advbench_query":       str,
      "tokens":               [str, ...],           # only the detection-mask tokens
      "scores": {                                   # signed logits; length == len(tokens)
        "<probe>":            [float, ...]
      },
      "sections": {                                 # [start, end) in detection-token coords
        "scratchpad":         [int, int],
        "response":           [int, int]            # absent if mask_kind == "scratchpad"
      },
      "metadata": {
        "mask_kind":          "scratchpad" | "response" | "both",
        "model":              str
      }
    }

Tokens are saved at the *detection-mask* granularity (matching the .npz
sample_id layout), so the scratchpad/response section boundaries are
computed within that subspace -- not against the full input token
sequence. This keeps the viewer simple and aligned to what the probe
actually saw.

Requires Apollo's deception_detection to be importable (pod-only).

Example:
    uv run python scripts/dev/save_per_token_scores.py \\
        --activations data/processed/headline_activations.npz \\
        --positives data/processed/headline_positives.jsonl \\
        --negatives data/processed/headline_negatives.jsonl \\
        --probes roleplaying followup instructed_pairs \\
        --output data/processed/per_token_scores.jsonl
"""

from __future__ import annotations

import argparse
import json
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

_DEFAULT_APOLLO_EXAMPLE_RESULTS = Path("/workspace/deception-detection/example_results")
_APOLLO_COMMIT = "f8ec4010e74927394709dffa22b97bdf8cd5a62f"


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


def _get_tokenized_attrs(tokenized: object) -> tuple[np.ndarray, np.ndarray]:
    detection_mask = tokenized.detection_mask.cpu().numpy()  # type: ignore[attr-defined]
    for attr in ("tokens", "input_ids", "tokens_ids"):
        if hasattr(tokenized, attr):
            tensor = getattr(tokenized, attr)
            return tensor.cpu().numpy(), detection_mask
    raise AttributeError(
        f"Apollo TokenizedDataset has no token-id attribute. "
        f"Available attrs: {[a for a in dir(tokenized) if not a.startswith('_')]}"
    )


#: Section-boundary markers, tried in order. The model is *supposed* to
#: wrap its deliberation in ``<SCRATCHPAD_REASONING>...</SCRATCHPAD_REASONING>``
#: tags per Redwood's helpful-only prompt, but only ~20% of generations
#: actually do this -- the rest deliberate inline and only emit a
#: commitment marker. Either commitment marker is a valid split point:
#: pre-marker is the deliberation (whether tag-wrapped or not), post-marker
#: is the response.
_SECTION_MARKERS = (
    "</SCRATCHPAD_REASONING>",
    "<request_fully_accepted/>",
    "<rejected/>",
)


def _split_scratchpad_response(
    detection_token_strs: list[str],
) -> dict[str, tuple[int, int]]:
    """Find the deliberation / response boundary within the detection-mask tokens.

    Tries three markers in order of preference:
    1. ``</SCRATCHPAD_REASONING>`` -- the prompt's expected wrapping; only
       ~20% of generations follow it.
    2. ``<request_fully_accepted/>`` -- AF compliance commitment.
    3. ``<rejected/>`` -- refusal commitment.

    Splits at the FIRST marker found (whichever appears earliest in the
    token stream). The section before the marker is labelled ``scratchpad``
    (a slight abuse of terminology -- it's the deliberation portion,
    whether tag-wrapped or not). The section at and after the marker is
    labelled ``response``.

    Returns ``{}`` when no marker is found at all (rare; ~1.5% in our
    headline cohort).
    """
    full = "".join(detection_token_strs)

    # Find the earliest occurring marker (any of the three).
    earliest_pos = -1
    earliest_marker = ""
    for marker in _SECTION_MARKERS:
        pos = full.find(marker)
        if pos < 0:
            continue
        if earliest_pos < 0 or pos < earliest_pos:
            earliest_pos = pos
            earliest_marker = marker
    if earliest_pos < 0:
        return {}

    # Map char position to token index. The boundary is the token AFTER the
    # last character of the marker (so the response section starts cleanly
    # with whatever comes next).
    target_char = earliest_pos + len(earliest_marker)
    running = 0
    boundary_tok = len(detection_token_strs)
    for i, tok in enumerate(detection_token_strs):
        running += len(tok)
        if running >= target_char:
            boundary_tok = i + 1
            break

    sections: dict[str, tuple[int, int]] = {}
    if boundary_tok > 0:
        sections["scratchpad"] = (0, boundary_tok)
    if boundary_tok < len(detection_token_strs):
        sections["response"] = (boundary_tok, len(detection_token_strs))
    return sections


def _label_for(transcript: LabelledTranscript) -> str:
    """Map LabelledTranscript.label to a viewer-friendly string.

    Default: ``"af"`` if label==1, ``"not_af"`` if label==0. If the
    transcript metadata carries a ``manual_label`` (from the manual review
    in ``results/af_pair_judgments.csv``), use that instead -- it
    distinguishes ``af`` from ``af_partial`` which the int label can't.
    """
    manual = transcript.transcript.metadata.get("manual_label")
    if manual in ("af", "af_partial", "not_af", "artifact"):
        return manual  # type: ignore[return-value]
    return "af" if transcript.label == 1 else "not_af"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--positives", type=Path, required=True)
    parser.add_argument("--negatives", type=Path, required=True)
    parser.add_argument(
        "--probes",
        nargs="+",
        default=["roleplaying", "followup", "instructed_pairs"],
        help="Apollo example_results sub-directories to load probes from.",
    )
    parser.add_argument(
        "--apollo-example-results",
        type=Path,
        default=_DEFAULT_APOLLO_EXAMPLE_RESULTS,
    )
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="Tokenizer to load. Must match the model that produced the activations.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Tokenizer max_length. Must match the headline extraction run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/per_token_scores.jsonl"),
    )
    parser.add_argument(
        "--legacy-shuffle",
        action="store_true",
        help=(
            "Set if the .npz was extracted before commit 810b96b (Apollo's "
            "auto-shuffle defaulted to on). The shuffle is replayed so "
            "tokenisation matches the saved activations."
        ),
    )
    args = parser.parse_args()

    print(f"Loading activations from {args.activations}")
    dataset = load_activation_dataset(args.activations)
    mask_kind_raw = dataset.metadata.get("detection_mask_kind", "both")
    if mask_kind_raw not in ("scratchpad", "response", "both"):
        raise ValueError(f"Unexpected detection_mask_kind: {mask_kind_raw!r}")
    mask_kind: DetectionMaskKind = mask_kind_raw  # type: ignore[assignment]
    n_samples = int(dataset.sample_id.max() + 1)
    print(
        f"  shape={dataset.activations.shape}, n_samples={n_samples}, "
        f"detection_mask_kind={mask_kind}"
    )

    print(f"Loading {len(args.probes)} probes from {args.apollo_example_results}")
    probe_results: dict[str, ProbeResult] = {}
    per_probe_scores: dict[str, np.ndarray] = {}
    for probe_name in args.probes:
        probe_result = _load_apollo_published_probe(
            apollo_example_results=args.apollo_example_results,
            scenario=probe_name,
            layer=args.layer,
        )
        probe_results[probe_name] = probe_result
        # predict_probe returns the *probability* of the positive class. For
        # the viewer we want the underlying signed logit so the colourmap
        # reads sign-meaningfully. Reconstruct it from the linear probe.
        scaled = probe_result.scaler.transform(dataset.activations)
        logits = scaled @ probe_result.probe.coef_[0] + probe_result.probe.intercept_[0]
        per_probe_scores[probe_name] = logits.astype(np.float32)
        # Also sanity-check against the probability path
        _ = predict_probe(probe_result, dataset.activations)
        print(
            f"  {probe_name:<20} logits min={logits.min():+.3f} max={logits.max():+.3f} "
            f"mean={logits.mean():+.3f}"
        )

    print("Re-tokenising the same dialogue dataset to recover token IDs")
    positives = load_labelled_transcripts(args.positives)
    negatives = load_labelled_transcripts(args.negatives)
    labelled = positives + negatives
    if len(labelled) != n_samples:
        raise ValueError(
            f"Positives+negatives JSONL has {len(labelled)} transcripts but the "
            f".npz was built from {n_samples} samples."
        )

    dialogue_dataset = build_af_dialogue_dataset(
        labelled, mask_kind=mask_kind, shuffle_upon_init=args.legacy_shuffle
    )

    if args.legacy_shuffle:
        np.random.seed(42)
        sid_to_input_idx = np.random.permutation(n_samples)
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
    n_detection = int(detection_mask.sum())
    if n_detection != dataset.activations.shape[0]:
        raise RuntimeError(
            f"Re-tokenised detection-mask count ({n_detection}) != saved per-token "
            f"activations ({dataset.activations.shape[0]}). Likely cause: --max-length "
            f"differs from the headline run, or positives/negatives don't match."
        )

    # Build the per-sample mapping: for each .npz token row, which
    # (sample, seq_pos) does it correspond to?
    flat_batch_idx = np.empty(dataset.activations.shape[0], dtype=np.int64)
    flat_seq_pos = np.empty(dataset.activations.shape[0], dtype=np.int64)
    cursor = 0
    for batch_idx in range(detection_mask.shape[0]):
        positions = np.where(detection_mask[batch_idx])[0]
        flat_batch_idx[cursor : cursor + positions.shape[0]] = batch_idx
        flat_seq_pos[cursor : cursor + positions.shape[0]] = positions
        cursor += positions.shape[0]
    if not np.array_equal(flat_batch_idx, dataset.sample_id):
        raise RuntimeError(
            "Reconstructed sample_id does not match saved sample_id. "
            "Most likely cause: the .npz was extracted before commit 810b96b "
            "(Apollo's auto-shuffle fix); rerun with --legacy-shuffle to "
            "replay the deterministic seed=42 permutation. "
            "Other possible causes: --max-length mismatch, or "
            "positives/negatives JSONL doesn't match the headline run."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with args.output.open("w", encoding="utf-8") as f:
        for sid in range(n_samples):
            sample_mask = dataset.sample_id == sid
            sample_seq_pos = flat_seq_pos[sample_mask]
            input_idx = int(sid_to_input_idx[sid])
            transcript = labelled[input_idx]

            # Decode each detection-mask token individually so we can
            # surface tokenisation choices in the viewer instead of
            # collapsing them by whole-string decoding.
            sample_token_ids = token_ids[sid, sample_seq_pos]
            token_strs = [
                tokenizer.decode([int(tid)], skip_special_tokens=False) for tid in sample_token_ids
            ]
            sections = (
                _split_scratchpad_response(token_strs)
                if mask_kind == "both"
                else {mask_kind: (0, len(token_strs))}
            )

            scores_payload = {
                probe_name: per_probe_scores[probe_name][sample_mask].tolist()
                for probe_name in args.probes
            }

            row = {
                "sample_id": int(sid),
                "input_idx": input_idx,
                "manual_label": _label_for(transcript),
                "is_free_tier_user": bool(
                    transcript.transcript.metadata.get("is_free_tier_user", False)
                ),
                "advbench_query": transcript.transcript.metadata.get("advbench_query", ""),
                "tokens": token_strs,
                "scores": scores_payload,
                "sections": {k: list(v) for k, v in sections.items()},
                "metadata": {
                    "mask_kind": mask_kind,
                    "model": args.model,
                    "apollo_commit": _APOLLO_COMMIT,
                },
            }
            f.write(json.dumps(row) + "\n")
            n_written += 1

    print(f"\nWrote {n_written} rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
