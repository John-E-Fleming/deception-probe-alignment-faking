"""Cross-probe summary: score all saved AF activations against all 3 Apollo probes.

Tests the cross-probe-family generalisation hypothesis. Apollo trained
three layer-22 deception probes on different instructed-deception
scenarios (roleplaying, insider_trading, sandbagging). The headline +
sensitivity findings so far cover only the roleplaying probe. This
script runs the OTHER two probes against the existing AF activation
.npz files so we can ask: does the compliance-mode-detection pattern
hold across the whole probe family, or is it specific to roleplaying?

Pure CPU; uses Apollo's published probes loaded from
``example_results/<scenario>/detector.pt``. Reuses every saved per-token
activation file -- no new GPU extraction is required.

Output: ``results/cross_probe_summary.json`` (full per-cell metrics with
bootstrap CIs) + a human-readable summary table printed to stdout, one
table per Apollo probe so the comparison is immediate.

Example (on the pod, default file list covers our standard 6 .npz files):
    uv run python scripts/dev/run_cross_probe_summary.py

To restrict to a subset:
    uv run python scripts/dev/run_cross_probe_summary.py \\
        --activations data/processed/headline_activations.npz \\
                      data/processed/sensitivity_af_partial_both.npz \\
        --scenarios roleplaying sandbagging
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from alignment_faking_probes.data.activation_store import load_activation_dataset
from alignment_faking_probes.probes.evaluation import ProbeMetrics, compute_metrics
from alignment_faking_probes.probes.training import (
    AggregationMethod,
    ProbeResult,
    aggregate_token_scores,
    predict_probe,
)

_DEFAULT_APOLLO_EXAMPLE_RESULTS = Path("/workspace/deception-detection/example_results")
_APOLLO_COMMIT = "f8ec4010e74927394709dffa22b97bdf8cd5a62f"
_LAYER = 22
_N_BOOT = 10_000

_APOLLO_SCENARIOS = ("roleplaying", "insider_trading", "sandbagging")
_AGGREGATIONS: tuple[AggregationMethod, ...] = ("mean", "max", "last")

# Default file list -- our six standard activation files. The script
# checks for existence and skips missing ones with a warning so the
# pod can still run partial summaries while data is being pulled.
_DEFAULT_FILES: tuple[str, ...] = (
    "data/processed/headline_activations.npz",
    "data/processed/headline_activations_scratchpad.npz",
    "data/processed/headline_activations_response.npz",
    "data/processed/sensitivity_af_partial_both.npz",
    "data/processed/sensitivity_af_partial_scratchpad.npz",
    "data/processed/sensitivity_af_partial_response.npz",
)


# TODO: same _load_apollo_published_probe duplication noted in the other
# four dev scripts. Refactor into alignment_faking_probes/probes/
# apollo_published.py and update all five call sites in one pass.
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
            f"Layer {layer} not in Apollo detector layers ({detector.layers}) "
            f"for scenario {scenario!r}."
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
    """Reduce per-token labels to per-sample labels in canonical 0..n-1 order."""
    n_samples = int(sample_id.max() + 1) if sample_id.size > 0 else 0
    per_sample = np.empty(n_samples, dtype=per_token_labels.dtype)
    for sid in range(n_samples):
        idx = np.where(sample_id == sid)[0][0]
        per_sample[sid] = per_token_labels[idx]
    return per_sample


def _short_name(path: Path) -> str:
    """Return a short identifier for a .npz file for the summary table."""
    name = path.stem  # strip .npz
    # Trim known prefixes to keep table columns short
    if name.startswith("headline_activations"):
        suffix = name[len("headline_activations") :].lstrip("_")
        return f"headline-{suffix or 'both'}"
    if name.startswith("sensitivity_af_partial_"):
        suffix = name[len("sensitivity_af_partial_") :]
        return f"sensitivity-{suffix}"
    return name


def _score_cell(
    npz_path: Path,
    scenario: str,
    aggregation: AggregationMethod,
    apollo_example_results: Path,
    n_boot: int,
) -> ProbeMetrics:
    """Score one (file, scenario, aggregation) cell."""
    dataset = load_activation_dataset(npz_path)
    probe_result = _load_apollo_published_probe(apollo_example_results, scenario, _LAYER)
    token_scores = predict_probe(probe_result, dataset.activations)
    sample_scores, _ = aggregate_token_scores(token_scores, dataset.sample_id, method=aggregation)
    sample_labels = _per_sample_labels(dataset.labels, dataset.sample_id)
    return compute_metrics(
        labels=sample_labels,
        scores=sample_scores,
        train_scenario=probe_result.train_scenario,
        eval_scenario=dataset.scenario,
        n_boot=n_boot,
    )


def _format_auroc_cell(m: ProbeMetrics | None) -> str:
    """Single-line AUROC + CI for the summary table."""
    if m is None:
        return "    -    "
    if m.auroc_ci is None:
        return f"{m.auroc:.3f}"
    return f"{m.auroc:.3f} [{m.auroc_ci[0]:.2f},{m.auroc_ci[1]:.2f}]"


def _print_summary(
    results: dict[str, dict[str, dict[str, ProbeMetrics]]],
    scenarios: list[str],
    aggregations: list[AggregationMethod],
    file_order: list[Path],
) -> None:
    """One table per Apollo probe; rows = .npz files, cols = aggregations."""
    print()
    print("=" * 100)
    print("CROSS-PROBE SUMMARY")
    print("=" * 100)
    print(
        "Each cell: AUROC [95% bootstrap CI]. Reading the tables:\n"
        "  - If high-AUROC vs refusal + at-chance vs af_partial pattern repeats across\n"
        "    all 3 probes, the 'compliance-mode detector, not AF-specific' framing\n"
        "    generalises to the whole Apollo instructed-deception probe family.\n"
        "  - If one probe shows a different pattern (e.g. sandbagging actually\n"
        "    discriminates af from af_partial), that's a more nuanced finding worth\n"
        "    digging into.\n"
    )

    for scenario in scenarios:
        print()
        print(f"--- Apollo probe: {scenario} ---")
        header = f"  {'.npz file':<32s}  " + "  ".join(f"{a:>17s}" for a in aggregations)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for npz in file_order:
            short = _short_name(npz)
            by_scenario = results.get(str(npz), {})
            by_agg = by_scenario.get(scenario, {})
            cells = [_format_auroc_cell(by_agg.get(a)) for a in aggregations]
            print(f"  {short:<32s}  " + "  ".join(f"{c:>17s}" for c in cells))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--activations",
        type=Path,
        nargs="+",
        default=[Path(p) for p in _DEFAULT_FILES],
        help=(
            "Activation .npz files to score. Default covers our standard six "
            "files (three headline mask scopes against refusal negatives + "
            "three against af_partial). Missing files are skipped with a warning."
        ),
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=list(_APOLLO_SCENARIOS),
        choices=list(_APOLLO_SCENARIOS),
        help="Apollo probe scenarios to test.",
    )
    parser.add_argument(
        "--aggregations",
        nargs="+",
        default=list(_AGGREGATIONS),
        choices=list(_AGGREGATIONS),
        help="Per-token -> per-sample aggregation methods.",
    )
    parser.add_argument(
        "--apollo-example-results",
        type=Path,
        default=_DEFAULT_APOLLO_EXAMPLE_RESULTS,
        help="Apollo's example_results/ directory containing detector.pt files.",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=_N_BOOT,
        help="Bootstrap resamples for the per-cell 95%% AUROC CI.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cross_probe_summary.json"),
        help="Where to write the full JSON of per-cell metrics.",
    )
    args = parser.parse_args()

    # Filter to existing files
    existing_files: list[Path] = []
    for p in args.activations:
        if p.exists():
            existing_files.append(p)
        else:
            print(f"WARN: skipping missing file {p}", file=sys.stderr)
    if not existing_files:
        print("ERROR: no input .npz files found.", file=sys.stderr)
        return 1

    # Verify all Apollo probes exist before doing any compute
    for scenario in args.scenarios:
        path = args.apollo_example_results / scenario / "detector.pt"
        if not path.exists():
            print(
                f"ERROR: Apollo probe for scenario {scenario!r} not found at {path}. "
                "Aborting before scoring to avoid partial results.",
                file=sys.stderr,
            )
            return 1

    print(
        f"Scoring {len(existing_files)} file(s) x {len(args.scenarios)} probe(s) x "
        f"{len(args.aggregations)} aggregation(s) = "
        f"{len(existing_files) * len(args.scenarios) * len(args.aggregations)} cells "
        f"with n_boot={args.n_boot}"
    )

    # results[npz_path_str][scenario][aggregation] = ProbeMetrics
    results: dict[str, dict[str, dict[str, ProbeMetrics]]] = {}
    for npz in existing_files:
        print(f"\n--- {npz.name} ---")
        results[str(npz)] = {}
        for scenario in args.scenarios:
            results[str(npz)][scenario] = {}
            for agg in args.aggregations:
                m = _score_cell(npz, scenario, agg, args.apollo_example_results, args.n_boot)
                results[str(npz)][scenario][agg] = m
                ci = (
                    f"[{m.auroc_ci[0]:.3f}, {m.auroc_ci[1]:.3f}]"
                    if m.auroc_ci is not None
                    else "n/a"
                )
                print(f"  probe={scenario:<16s} agg={agg:<5s} AUROC={m.auroc:.4f} {ci}")

    # Serialise: ProbeMetrics -> dict
    serialisable: dict[str, object] = {
        "config": {
            "scenarios": args.scenarios,
            "aggregations": args.aggregations,
            "layer": _LAYER,
            "n_boot": args.n_boot,
            "apollo_commit": _APOLLO_COMMIT,
        },
        "results": {},
    }
    for npz_str, by_scenario in results.items():
        serialisable["results"][npz_str] = {  # type: ignore[index]
            scenario: {
                agg: {
                    "auroc": m.auroc,
                    "auroc_ci": list(m.auroc_ci) if m.auroc_ci is not None else None,
                    "tpr_at_5pct_fpr": m.tpr_at_5pct_fpr,
                    "tpr_at_5pct_fpr_ci": (
                        list(m.tpr_at_5pct_fpr_ci) if m.tpr_at_5pct_fpr_ci is not None else None
                    ),
                    "tpr_at_1pct_fpr": m.tpr_at_1pct_fpr,
                    "tpr_at_1pct_fpr_ci": (
                        list(m.tpr_at_1pct_fpr_ci) if m.tpr_at_1pct_fpr_ci is not None else None
                    ),
                    "n_positive": m.n_positive,
                    "n_negative": m.n_negative,
                }
                for agg, m in by_agg.items()
            }
            for scenario, by_agg in by_scenario.items()
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(serialisable, indent=2))
    print(f"\nWrote full per-cell metrics -> {args.output}")

    _print_summary(results, args.scenarios, args.aggregations, existing_files)
    return 0


if __name__ == "__main__":
    sys.exit(main())
