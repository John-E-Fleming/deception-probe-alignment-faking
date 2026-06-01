"""Headline experiment: Apollo roleplaying probe -> naturally-emerging AF transcripts.

This is the single cell that motivates Blog Post 4. We take Apollo's
published roleplaying probe (trained on instructed deception), evaluate it
on Llama-3.3-70B alignment-faking transcripts, and report per-sample
AUROC. Either result direction is a contribution:

- High AUROC => instructed-deception probes transfer to naturally-emerging
  deception. Major positive result for probe-based monitoring.
- Low AUROC => they don't, confirming the Kirch et al. off-policy
  prediction at the hardest distribution shift.

Pipeline:
1. Load AF transcripts labelled by ``02_label_af_transcripts.py``.
2. Filter to ``included=True`` and build an Apollo DialogueDataset with
   the chosen detection mask (default: scratchpad + response).
3. Extract per-token activations at Apollo's roleplaying layer (22).
4. Load Apollo's published roleplaying probe weights from
   ``example_results/`` (NEVER retrained -- replication is a precondition,
   not an open question).
5. ``predict_probe`` -> ``aggregate_token_scores(mean)`` for per-sample scores.
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from alignment_faking_probes import ALIGNMENT_FAKING
from alignment_faking_probes.data.activation_extraction import (
    DetectionMaskKind,
    ExtractionConfig,
    build_af_dialogue_dataset,
    extract_activations,
)
from alignment_faking_probes.data.activation_store import save_activation_dataset
from alignment_faking_probes.data.labeling import (
    LabelledTranscript,
    filter_included,
    load_labelled_transcripts,
)
from alignment_faking_probes.data.model_loading import load_model_and_tokenizer
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

#: Number of stratified-bootstrap resamples for the AUROC + TPR-at-FPR CIs.
#: 10,000 is the conventional choice for percentile-bootstrap CIs at this
#: precision and is cheap once the heavy lifting (activation extraction)
#: is done.
_N_BOOT = 10_000


def _load_curated_pair(positives_path: Path, negatives_path: Path) -> list[LabelledTranscript]:
    """Load the manually-curated positive/negative LabelledTranscripts.

    Sanity-checks that labels match the file they came from — protects
    against accidentally swapping the two flags on the command line.
    Returns positives followed by negatives so the sample-index order is
    stable and predictable.
    """
    if not positives_path.exists():
        raise FileNotFoundError(f"Positives file not found: {positives_path}")
    if not negatives_path.exists():
        raise FileNotFoundError(f"Negatives file not found: {negatives_path}")
    positives = load_labelled_transcripts(positives_path)
    negatives = load_labelled_transcripts(negatives_path)
    bad_pos = [i for i, item in enumerate(positives) if item.label != 1]
    bad_neg = [i for i, item in enumerate(negatives) if item.label != 0]
    if bad_pos:
        raise ValueError(
            f"--positives at {positives_path} contains rows with label != 1 "
            f"at indices {bad_pos[:5]}{'...' if len(bad_pos) > 5 else ''} -- "
            "did you swap --positives and --negatives?"
        )
    if bad_neg:
        raise ValueError(
            f"--negatives at {negatives_path} contains rows with label != 0 "
            f"at indices {bad_neg[:5]}{'...' if len(bad_neg) > 5 else ''} -- "
            "did you swap --positives and --negatives?"
        )
    return positives + negatives


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
        default=None,
        help=(
            "LabelledTranscript JSONL from 02_label_af_transcripts.py. Auto-filtered "
            "to rows with included=True. Mutually exclusive with --positives/--negatives."
        ),
    )
    parser.add_argument(
        "--positives",
        type=Path,
        default=None,
        help=(
            "Curated LabelledTranscript JSONL for the positive (AF) class -- typically "
            "the manually-validated `af` transcripts from build_experiment_dataset.py. "
            "Requires --negatives. Bypasses the classifier-confidence filter."
        ),
    )
    parser.add_argument(
        "--negatives",
        type=Path,
        default=None,
        help=("Curated LabelledTranscript JSONL for the negative class. Requires --positives."),
    )
    parser.add_argument(
        "--save-activations",
        type=Path,
        default=None,
        help=(
            "Optional .npz path. When set, persist the per-token activations + "
            "sample_id + labels for downstream reuse (e.g. learning-curve training "
            "without re-extracting). Sidecar JSON is written alongside."
        ),
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
    parser.add_argument(
        "--adapter",
        default=None,
        help=(
            "Optional HF repo ID for a PEFT LoRA adapter to load over the base model. "
            "For AF transfer evaluation, the recommended adapter is "
            "'jplhughes2/llama-3.3-70b-af-synthetic-docs-only-more-data-r-epochs'."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument(
        "--aggregation",
        choices=["mean", "max", "last"],
        default="mean",
        help="Per-token -> per-sample score aggregation. Mean matches Apollo's default.",
    )
    args = parser.parse_args()

    using_curated = args.positives is not None or args.negatives is not None
    if using_curated:
        if args.positives is None or args.negatives is None:
            raise ValueError("--positives and --negatives must be supplied together.")
        if args.labelled is not None:
            raise ValueError(
                "--labelled is mutually exclusive with --positives/--negatives. "
                "Pick one input mode."
            )
    elif args.labelled is None:
        raise ValueError("Must supply either --labelled or both --positives and --negatives.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if using_curated:
        print(
            f"Loading curated positive/negative LabelledTranscripts from "
            f"{args.positives} and {args.negatives}"
        )
        labelled = _load_curated_pair(args.positives, args.negatives)
        labelled_all = labelled  # nothing was excluded; everything is curated
        labelled_source = f"{args.positives}|{args.negatives}"
    else:
        print(f"Loading labelled AF transcripts from {args.labelled}")
        assert args.labelled is not None  # narrowed by branch above
        labelled_all = load_labelled_transcripts(args.labelled)
        labelled = filter_included(labelled_all)
        labelled_source = str(args.labelled)
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
    model, tokenizer = load_model_and_tokenizer(
        args.model, args.quantisation, adapter_repo_id=args.adapter
    )
    if args.adapter is not None:
        print(f"Applied LoRA adapter: {args.adapter}")

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
        scenario=ALIGNMENT_FAKING,
        apollo_commit=_APOLLO_COMMIT,
    )
    eval_dataset = datasets[args.layer]
    print(
        f"  Per-token activations: shape={eval_dataset.activations.shape}, "
        f"unique samples={eval_dataset.sample_id.max() + 1}"
    )

    if args.save_activations is not None:
        args.save_activations.parent.mkdir(parents=True, exist_ok=True)
        save_activation_dataset(eval_dataset, args.save_activations)
        print(f"Saved activation dataset -> {args.save_activations} (+ .json sidecar)")

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
        eval_scenario=ALIGNMENT_FAKING,
        n_boot=_N_BOOT,
    )

    def _fmt_ci(ci: tuple[float, float] | None) -> str:
        if ci is None:
            return "n/a"
        return f"[{ci[0]:.4f}, {ci[1]:.4f}]"

    print("\n=== Headline transfer cell ===")
    print(f"  Train scenario: {metrics.train_scenario}")
    print(f"  Eval scenario:  {metrics.eval_scenario}")
    print(f"  AUROC:          {metrics.auroc:.4f}  95% CI {_fmt_ci(metrics.auroc_ci)}")
    print(
        f"  TPR @ 1% FPR:   {metrics.tpr_at_1pct_fpr:.4f}  95% CI "
        f"{_fmt_ci(metrics.tpr_at_1pct_fpr_ci)}"
    )
    print(
        f"  TPR @ 5% FPR:   {metrics.tpr_at_5pct_fpr:.4f}  95% CI "
        f"{_fmt_ci(metrics.tpr_at_5pct_fpr_ci)}"
    )
    print(f"  n_positive:     {metrics.n_positive}")
    print(f"  n_negative:     {metrics.n_negative}")
    print(f"  n_boot:         {metrics.n_boot}")

    result_payload = {
        "metrics": {
            "auroc": metrics.auroc,
            "auroc_ci": list(metrics.auroc_ci) if metrics.auroc_ci is not None else None,
            "tpr_at_1pct_fpr": metrics.tpr_at_1pct_fpr,
            "tpr_at_1pct_fpr_ci": (
                list(metrics.tpr_at_1pct_fpr_ci) if metrics.tpr_at_1pct_fpr_ci is not None else None
            ),
            "tpr_at_5pct_fpr": metrics.tpr_at_5pct_fpr,
            "tpr_at_5pct_fpr_ci": (
                list(metrics.tpr_at_5pct_fpr_ci) if metrics.tpr_at_5pct_fpr_ci is not None else None
            ),
            "n_positive": metrics.n_positive,
            "n_negative": metrics.n_negative,
            "n_boot": metrics.n_boot,
            "train_scenario": metrics.train_scenario,
            "eval_scenario": metrics.eval_scenario,
        },
        "config": {
            "model": args.model,
            "adapter": args.adapter,
            "quantisation": args.quantisation,
            "layer": args.layer,
            "detection_mask_kind": mask_kind,
            "aggregation": args.aggregation,
            "probe_scenario": args.probe_scenario,
            "apollo_commit": _APOLLO_COMMIT,
            "labelled_transcripts": labelled_source,
            "positives": str(args.positives) if args.positives is not None else None,
            "negatives": str(args.negatives) if args.negatives is not None else None,
            "n_total_labelled": len(labelled_all),
            "n_included": len(labelled),
            "save_activations": (
                str(args.save_activations) if args.save_activations is not None else None
            ),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    result_path = args.output_dir / "headline_apollo_to_af.json"
    result_path.write_text(json.dumps(result_payload, indent=2))
    print(f"\nWrote metrics -> {result_path}")

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
    print(f"Wrote ROC plot -> {plot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
