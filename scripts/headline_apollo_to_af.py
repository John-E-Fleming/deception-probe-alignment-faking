"""Headline experiment: Apollo roleplaying probe → naturally-emerging AF transcripts.

This is the single cell that motivates Blog Post 4. We take Apollo's
published roleplaying probe (trained on instructed deception), evaluate it
on Llama-3.3-70B alignment-faking transcripts, and report per-sample
AUROC. Either result direction is a contribution:

- High AUROC ⇒ instructed-deception probes transfer to naturally-emerging
  deception. Major positive result for probe-based monitoring.
- Low AUROC ⇒ they don't, confirming the Kirch et al. off-policy
  prediction at the hardest distribution shift.

Pipeline:
1. Load AF transcripts labelled by ``02_label_af_transcripts.py``.
2. Filter to ``included=True`` and build an Apollo DialogueDataset with
   the chosen detection mask (default: scratchpad + response).
3. Extract per-token activations at Apollo's roleplaying layer (22).
4. Load Apollo's published roleplaying probe weights from
   ``example_results/`` (NEVER retrained — replication is a precondition,
   not an open question).
5. ``predict_probe`` → ``aggregate_token_scores(mean)`` for per-sample scores.
6. ``compute_metrics`` against AF classifier labels.
7. Write ``results/headline_apollo_to_af.json`` and an ROC PNG.

Run on RunPod after activations are extractable.

Example:
    uv run python scripts/headline_apollo_to_af.py \\
        --labelled data/labelled/af_labelled.jsonl \\
        --output-dir results/
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from alignment_faking_probes.data.activation_extraction import (
    DetectionMaskKind,
    ExtractionConfig,
    build_af_dialogue_dataset,
    extract_activations,
)
from alignment_faking_probes.data.labeling import (
    filter_included,
    load_labelled_transcripts,
)
from alignment_faking_probes.probes.evaluation import compute_metrics, plot_roc_curve
from alignment_faking_probes.probes.training import (
    ProbeResult,
    aggregate_token_scores,
    predict_probe,
)

#: Apollo's roleplaying probe is trained on layer 22 of Llama-3.3-70B.
#: Hardcoded here because it's a property of Apollo's published artefact,
#: not a hyperparameter we tune.
_APOLLO_ROLEPLAYING_LAYER = 22

#: Default location of Apollo's example_results/ on RunPod (matches
#: scripts/00_runpod_setup.sh layout).
_DEFAULT_APOLLO_EXAMPLE_RESULTS = Path("/workspace/deception-detection/example_results")

#: Apollo's pinned commit (kept in sync with pyproject.toml + 00_runpod_setup.sh).
#: Logged in the result JSON for provenance — every transfer experiment must
#: record which Apollo revision its probe weights came from.
_APOLLO_COMMIT = "f8ec4010e74927394709dffa22b97bdf8cd5a62f"


def _load_model_and_tokenizer(model_name: str, quantisation: str):  # type: ignore[no-untyped-def]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if quantisation == "4bit":
        quant_config = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
        )
    else:  # bf16
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    return model, tokenizer


def _load_apollo_published_probe(
    apollo_example_results: Path, scenario: str, layer: int
) -> ProbeResult:
    """Wrap Apollo's published detector as our ProbeResult.

    Same construction as ``check_apollo_format.py``: slice the layer's
    direction + scaler stats out of Apollo's detector and rebuild a
    sklearn ``LogisticRegression`` (no intercept — Apollo's detector
    doesn't have one) plus a ``StandardScaler``.
    """
    from deception_detection.detectors import LogisticRegressionDetector

    detector_path = apollo_example_results / scenario / "detector.pt"
    if not detector_path.exists():
        raise FileNotFoundError(
            f"Apollo published probe not found at {detector_path}. "
            "Check --apollo-example-results points to deception-detection/example_results."
        )
    detector = LogisticRegressionDetector.load(detector_path)
    if layer not in detector.layers:
        raise ValueError(
            f"Layer {layer} not in Apollo detector layers ({detector.layers}). "
            "Roleplaying detector is trained at layer 22 by default."
        )
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labelled",
        type=Path,
        default=Path("data/labelled/af_labelled.jsonl"),
        help="LabelledTranscript JSONL from 02_label_af_transcripts.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Where to write headline_apollo_to_af.json and the ROC plot.",
    )
    parser.add_argument(
        "--probe-scenario",
        default="roleplaying",
        help="Subdir of example_results/ holding the probe to evaluate.",
    )
    parser.add_argument(
        "--apollo-example-results",
        type=Path,
        default=_DEFAULT_APOLLO_EXAMPLE_RESULTS,
    )
    parser.add_argument("--layer", type=int, default=_APOLLO_ROLEPLAYING_LAYER)
    parser.add_argument(
        "--detection-mask",
        choices=["scratchpad", "response", "both"],
        default="both",
        help="Which tokens count as detection positions in AF transcripts.",
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--quantisation", choices=["4bit", "bf16"], default="4bit")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument(
        "--aggregation",
        choices=["mean", "max", "last"],
        default="mean",
        help="Per-token → per-sample score aggregation. Mean matches Apollo's default.",
    )
    args = parser.parse_args()

    if not args.labelled.exists():
        raise FileNotFoundError(f"Labelled transcripts file not found: {args.labelled}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading labelled AF transcripts from {args.labelled}")
    labelled_all = load_labelled_transcripts(args.labelled)
    labelled = filter_included(labelled_all)
    n_pos = sum(1 for item in labelled if item.label == 1)
    n_neg = sum(1 for item in labelled if item.label == 0)
    print(
        f"  Loaded {len(labelled_all)} transcripts; "
        f"{len(labelled)} included ({n_pos} AF, {n_neg} non-AF)."
    )
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"Need both classes for AUROC; got {n_pos} positive / {n_neg} negative. "
            "Check the classifier output and the --threshold used in step 02."
        )

    mask_kind: DetectionMaskKind = args.detection_mask
    print(f"Building Apollo DialogueDataset (mask_kind={mask_kind})")
    dialogue_dataset = build_af_dialogue_dataset(labelled, mask_kind=mask_kind)

    print(f"Loading {args.model} (quantisation={args.quantisation})")
    model, tokenizer = _load_model_and_tokenizer(args.model, args.quantisation)

    config = ExtractionConfig(
        model_name=args.model,
        layers=[args.layer],
        batch_size=args.batch_size,
        max_length=args.max_length,
        detection_mask_kind=mask_kind,
    )
    print(f"Extracting per-token activations at layer {args.layer}")
    datasets = extract_activations(
        dialogue_dataset=dialogue_dataset,
        config=config,
        model=model,
        tokenizer=tokenizer,
        scenario="af_helpful_only",
        apollo_commit=_APOLLO_COMMIT,
    )
    eval_dataset = datasets[args.layer]
    print(
        f"  Per-token activations: shape={eval_dataset.activations.shape}, "
        f"unique samples={eval_dataset.sample_id.max() + 1}"
    )

    print(f"Loading Apollo's published probe (scenario={args.probe_scenario})")
    probe_result = _load_apollo_published_probe(
        apollo_example_results=args.apollo_example_results,
        scenario=args.probe_scenario,
        layer=args.layer,
    )

    print(f"Scoring + aggregating (method={args.aggregation})")
    token_scores = predict_probe(probe_result, eval_dataset.activations)
    sample_scores, sample_ids = aggregate_token_scores(
        token_scores, eval_dataset.sample_id, method=args.aggregation
    )
    # Sample labels are per-sample in the dialogue dataset construction order.
    # `aggregate_token_scores` returns sample_ids in ascending order, which
    # matches the order labels were appended in.
    from deception_detection.types import Label

    sample_labels = np.array(
        [1 if lab == Label.DECEPTIVE else 0 for lab in dialogue_dataset.labels],
        dtype=np.int64,
    )
    if sample_labels.shape != sample_scores.shape:
        raise RuntimeError(
            f"Per-sample label/score shape mismatch: "
            f"{sample_labels.shape} vs {sample_scores.shape}. "
            "Some samples may have had zero detection tokens after masking."
        )

    metrics = compute_metrics(
        labels=sample_labels,
        scores=sample_scores,
        train_scenario=probe_result.train_scenario,
        eval_scenario="af_helpful_only",
    )

    print("\n=== Headline transfer cell ===")
    print(f"  Train scenario: {metrics.train_scenario}")
    print(f"  Eval scenario:  {metrics.eval_scenario}")
    print(f"  AUROC:          {metrics.auroc:.4f}")
    print(f"  TPR @ 1% FPR:   {metrics.tpr_at_1pct_fpr:.4f}")
    print(f"  TPR @ 5% FPR:   {metrics.tpr_at_5pct_fpr:.4f}")
    print(f"  n_positive:     {metrics.n_positive}")
    print(f"  n_negative:     {metrics.n_negative}")

    result_payload = {
        "metrics": {
            "auroc": metrics.auroc,
            "tpr_at_1pct_fpr": metrics.tpr_at_1pct_fpr,
            "tpr_at_5pct_fpr": metrics.tpr_at_5pct_fpr,
            "n_positive": metrics.n_positive,
            "n_negative": metrics.n_negative,
            "train_scenario": metrics.train_scenario,
            "eval_scenario": metrics.eval_scenario,
        },
        "config": {
            "model": args.model,
            "quantisation": args.quantisation,
            "layer": args.layer,
            "detection_mask_kind": mask_kind,
            "aggregation": args.aggregation,
            "probe_scenario": args.probe_scenario,
            "apollo_commit": _APOLLO_COMMIT,
            "labelled_transcripts": str(args.labelled),
            "n_total_labelled": len(labelled_all),
            "n_included": len(labelled),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    result_path = args.output_dir / "headline_apollo_to_af.json"
    result_path.write_text(json.dumps(result_payload, indent=2))
    print(f"\nWrote metrics → {result_path}")

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_roc_curve(
        labels=sample_labels,
        scores=sample_scores,
        label=f"Apollo {args.probe_scenario} probe → AF",
        ax=ax,
    )
    ax.set_title("Headline transfer: Apollo roleplaying → naturally-emerging AF")
    ax.legend(loc="lower right")
    plot_path = args.output_dir / "headline_apollo_to_af_roc.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote ROC plot → {plot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
