"""Re-aggregate a saved activations .npz under different pooling methods.

Pure CPU work -- loads a previously-saved per-token ActivationDataset,
re-applies Apollo's published probe, and reports per-sample AUROC plus
stratified-bootstrap 95% CIs for each of mean / max / last aggregation.

This is the cheap diagnostic for "how sensitive is the headline AUROC to
the per-token-to-per-sample pooling choice?" Apollo's default is mean
(matches their ``MeanPromptScorer``); max and last are the canonical
ablations.

Re-uses Apollo's roleplaying detector loading from
``scripts/headline_apollo_to_af.py`` (the helper is private but the
function body is small and intentionally kept identical to both that
script and ``scripts/check_apollo_format.py``; we duplicate it here as
a third copy with a TODO to consolidate -- consolidation is a separate
refactor, not in scope for this ablation pass).

Run on the pod (or anywhere Apollo is installed). Inputs are the .npz
files produced by ``headline_apollo_to_af.py --save-activations``.

Example:
    uv run python scripts/dev/reaggregate_ablations.py \\
        --activations data/processed/headline_activations.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from alignment_faking_probes import ALIGNMENT_FAKING
from alignment_faking_probes.data.activation_store import load_activation_dataset
from alignment_faking_probes.probes.evaluation import ProbeMetrics, compute_metrics
from alignment_faking_probes.probes.training import (
    AggregationMethod,
    ProbeResult,
    aggregate_token_scores,
    predict_probe,
)

#: Apollo's roleplaying probe layer. Hardcoded here because it's a property
#: of Apollo's published artefact, not a tunable.
_APOLLO_ROLEPLAYING_LAYER = 22

#: Default location of Apollo's example_results/ on RunPod.
_DEFAULT_APOLLO_EXAMPLE_RESULTS = Path("/workspace/deception-detection/example_results")

#: Apollo's pinned commit (kept in sync with pyproject.toml).
_APOLLO_COMMIT = "f8ec4010e74927394709dffa22b97bdf8cd5a62f"

#: Bootstrap resamples for AUROC + TPR-at-FPR CIs. Matches the headline.
_N_BOOT = 10_000

_AGGREGATIONS: tuple[AggregationMethod, ...] = ("mean", "max", "last")


# TODO: consolidate -- this function is currently duplicated across
# headline_apollo_to_af.py, check_apollo_format.py, and now this script.
# Worth factoring into alignment_faking_probes/probes/apollo_published.py
# in a follow-up refactor.
def _load_apollo_published_probe(
    apollo_example_results: Path, scenario: str, layer: int
) -> ProbeResult:
    """Wrap Apollo's published detector as our ProbeResult.

    Slices the layer's direction + scaler stats out of Apollo's detector
    and rebuilds a sklearn ``LogisticRegression`` (no intercept) + a
    ``StandardScaler``.
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


def _per_sample_labels(per_token_labels: np.ndarray, sample_id: np.ndarray) -> np.ndarray:
    """Reduce per-token labels to per-sample labels in canonical 0..n-1 order.

    ActivationDataset.__post_init__ already validates per-sample label
    consistency, so we can safely take the first label seen for each
    sample id without checking the rest.
    """
    n_samples = int(sample_id.max() + 1) if sample_id.size > 0 else 0
    per_sample = np.empty(n_samples, dtype=per_token_labels.dtype)
    for sid in range(n_samples):
        idx = np.where(sample_id == sid)[0][0]
        per_sample[sid] = per_token_labels[idx]
    return per_sample


def _format_ci(ci: tuple[float, float] | None) -> str:
    if ci is None:
        return "n/a"
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def _print_table(rows: list[tuple[str, ProbeMetrics]]) -> None:
    """Print a compact comparison table across aggregation methods."""
    header = (
        f"{'method':<8}  {'AUROC':>7}  {'AUROC 95% CI':>20}  "
        f"{'TPR@5%FPR':>10}  {'TPR@5% CI':>20}  "
        f"{'TPR@1%FPR':>10}  {'TPR@1% CI':>20}"
    )
    print(header)
    print("-" * len(header))
    for method, m in rows:
        print(
            f"{method:<8}  "
            f"{m.auroc:>7.4f}  {_format_ci(m.auroc_ci):>20}  "
            f"{m.tpr_at_5pct_fpr:>10.4f}  {_format_ci(m.tpr_at_5pct_fpr_ci):>20}  "
            f"{m.tpr_at_1pct_fpr:>10.4f}  {_format_ci(m.tpr_at_1pct_fpr_ci):>20}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--activations",
        type=Path,
        required=True,
        help=(
            "Path to a saved per-token ActivationDataset .npz "
            "(from headline_apollo_to_af.py --save-activations)."
        ),
    )
    parser.add_argument(
        "--apollo-example-results",
        type=Path,
        default=_DEFAULT_APOLLO_EXAMPLE_RESULTS,
        help="Apollo's example_results/ directory containing detector.pt files.",
    )
    parser.add_argument(
        "--probe-scenario",
        default="roleplaying",
        help="Subdir of example_results/ holding the probe to evaluate.",
    )
    parser.add_argument("--layer", type=int, default=_APOLLO_ROLEPLAYING_LAYER)
    parser.add_argument(
        "--n-boot",
        type=int,
        default=_N_BOOT,
        help="Bootstrap resamples for AUROC + TPR-at-FPR CIs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Bootstrap seed (reproducibility).",
    )
    args = parser.parse_args()

    print(f"Loading activations from {args.activations}")
    dataset = load_activation_dataset(args.activations)
    n_samples = int(dataset.sample_id.max() + 1)
    print(
        f"  shape={dataset.activations.shape}, n_samples={n_samples} "
        f"(n_positive={dataset.n_positive}, n_negative={dataset.n_negative}), "
        f"scenario={dataset.scenario}, layer={dataset.layer}"
    )

    print(f"Loading Apollo's published probe (scenario={args.probe_scenario})")
    probe_result = _load_apollo_published_probe(
        apollo_example_results=args.apollo_example_results,
        scenario=args.probe_scenario,
        layer=args.layer,
    )

    print("Scoring per-token activations (probe is fixed; pure CPU)")
    token_scores = predict_probe(probe_result, dataset.activations)

    sample_labels = _per_sample_labels(dataset.labels, dataset.sample_id)

    eval_scenario_for_metrics = (
        ALIGNMENT_FAKING if dataset.scenario == ALIGNMENT_FAKING else dataset.scenario
    )

    rows: list[tuple[str, ProbeMetrics]] = []
    for method in _AGGREGATIONS:
        sample_scores, sample_ids = aggregate_token_scores(
            token_scores, dataset.sample_id, method=method
        )
        assert sample_scores.shape == sample_labels.shape, (
            f"Shape mismatch between scores and labels for method={method}: "
            f"{sample_scores.shape} vs {sample_labels.shape}"
        )
        metrics = compute_metrics(
            labels=sample_labels,
            scores=sample_scores,
            train_scenario=probe_result.train_scenario,
            eval_scenario=eval_scenario_for_metrics,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        rows.append((method, metrics))

    print(
        f"\n=== Aggregation ablation (activations={args.activations.name}, "
        f"n_pos={dataset.n_positive}, n_neg={dataset.n_negative}, n_boot={args.n_boot}) ==="
    )
    _print_table(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
