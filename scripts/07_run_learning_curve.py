"""Stage 2 learning curve: does adding N AF examples lift the probe's
AUROC on AF reasoning?

The headline experiment established that Apollo's roleplaying probe
is scratchpad-blind (AUROC 0.60 [0.43, 0.77] under mean aggregation,
CI crosses chance, n_pos=19 af vs n_neg=50 refusals). The af_partial
sensitivity test sharpened this: the probe is detecting compliance-
mode output, not AF-specific reasoning. That leaves a specific open
question for probe-based AF monitoring:

    **Can a fresh probe trained with even a handful of AF examples
    detect alignment-faking reasoning in the scratchpad?**

This script answers it via a learning-curve experiment. For each
N in --n-list, K independent probes are trained on N AF positive
scratchpad activations + all available refusal-negative scratchpad
activations (held-out test set excluded). Each probe is evaluated on
the same fixed held-out test set; we report per-N mean AUROC ± std
across the K seeds, plus a 95% bootstrap CI on each individual seed.

The headline's Apollo-probe scratchpad AUROC (0.60) is drawn on the
plot as a reference line — the "what was achievable without retraining"
baseline.

**Mask scope is parameterised but the project's bookend uses
--mask scratchpad.** That's the cleanest test because it directly
attacks the limitation we identified. Other masks are supported for
flexibility (e.g. response or both, optionally including Apollo's
roleplaying contrast pairs in training via --apollo-activations).

Apollo's prewritten roleplaying data has **no scratchpad tokens** — so
under --mask scratchpad the --apollo-activations flag is ignored with
a warning. Pure-AF training is the only defensible setup at that
mask scope.

Example (the project's actual experiment):
    uv run python scripts/07_run_learning_curve.py \\
        --combined-activations data/processed/headline_activations_scratchpad.npz \\
        --mask scratchpad \\
        --output-dir results/learning_curve/

Run on the pod (Apollo's `deception_detection` is needed because the
saved .npz files reference it through ProbeResult metadata; the actual
training is sklearn-only). For local re-plotting from saved JSON, use
``--replot-from``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from alignment_faking_probes import validate_scenario
from alignment_faking_probes.data.activation_store import (
    ActivationDataset,
    load_activation_dataset,
)
from alignment_faking_probes.probes.evaluation import (
    ProbeMetrics,
    compute_metrics,
)
from alignment_faking_probes.probes.training import (
    AggregationMethod,
    aggregate_token_scores,
    predict_probe,
    train_probe,
)

MaskScope = Literal["scratchpad", "response", "both"]
_VALID_MASKS: tuple[MaskScope, ...] = ("scratchpad", "response", "both")

#: Apollo's published-probe scratchpad AUROC against refusal negatives.
#: Used as the N=0 reference line on the learning-curve plot under
#: --mask scratchpad. Lives in headline_apollo_to_af.json from the
#: earlier headline run; hardcoded here so the plot is reproducible
#: without re-loading that JSON.
_APOLLO_BASELINE_BY_MASK: dict[MaskScope, float] = {
    "scratchpad": 0.6032,
    "response": 0.7116,
    "both": 0.9221,
}

_DEFAULT_N_LIST: tuple[int, ...] = (3, 5, 10, 14)
_DEFAULT_K_SEEDS = 10
_DEFAULT_N_TEST_POSITIVES = 5
_N_BOOT = 10_000


def _select_samples(dataset: ActivationDataset, sample_ids: list[int]) -> ActivationDataset:
    """Return a new ActivationDataset containing only the given sample IDs.

    Sample IDs are remapped to a contiguous range 0..len(sample_ids)-1 so
    the ActivationDataset invariants (contiguous sample_id) hold on the
    returned subset.

    Args:
        dataset: Source dataset with arbitrary subset of original sample IDs.
        sample_ids: Sample IDs (in the dataset's original numbering) to keep.
            Order is preserved -- the output's sample_id=0 corresponds to
            sample_ids[0], etc.

    Returns:
        A new ActivationDataset with renumbered sample IDs and recounted
        n_positive / n_negative.
    """
    if not sample_ids:
        raise ValueError("sample_ids must be non-empty")

    # Build the new activation rows, in the order specified by sample_ids
    activation_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    new_sample_id_chunks: list[np.ndarray] = []
    sample_labels: list[int] = []

    for new_sid, old_sid in enumerate(sample_ids):
        mask = dataset.sample_id == old_sid
        if not mask.any():
            raise ValueError(f"Sample ID {old_sid} not present in dataset")
        activation_chunks.append(dataset.activations[mask])
        label_chunks.append(dataset.labels[mask])
        n_tokens = int(mask.sum())
        new_sample_id_chunks.append(np.full(n_tokens, new_sid, dtype=dataset.sample_id.dtype))
        sample_labels.append(int(dataset.labels[mask][0]))

    activations = np.concatenate(activation_chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0)
    sample_id = np.concatenate(new_sample_id_chunks, axis=0)

    n_pos = sum(1 for label in sample_labels if label == 1)
    n_neg = sum(1 for label in sample_labels if label == 0)

    return ActivationDataset(
        activations=activations,
        labels=labels,
        sample_id=sample_id,
        scenario=dataset.scenario,
        layer=dataset.layer,
        model_name=dataset.model_name,
        n_positive=n_pos,
        n_negative=n_neg,
        metadata=dataset.metadata.copy(),
    )


def _combine_datasets(datasets: list[ActivationDataset]) -> ActivationDataset:
    """Concatenate multiple ActivationDatasets, renumbering sample IDs.

    All inputs must have identical ``scenario``, ``layer``, and ``model_name``.
    Metadata is merged shallowly with later datasets overriding earlier ones.

    Raises:
        ValueError: If the input list is empty, or if scenario / layer /
            model_name disagree across inputs.
    """
    if not datasets:
        raise ValueError("Cannot combine empty list of datasets")

    first = datasets[0]
    for d in datasets[1:]:
        if d.scenario != first.scenario:
            raise ValueError(f"scenario mismatch: {d.scenario!r} vs {first.scenario!r}")
        if d.layer != first.layer:
            raise ValueError(f"layer mismatch: {d.layer} vs {first.layer}")
        if d.model_name != first.model_name:
            raise ValueError(f"model_name mismatch: {d.model_name!r} vs {first.model_name!r}")

    activation_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    sample_id_chunks: list[np.ndarray] = []
    offset = 0
    n_pos_total = 0
    n_neg_total = 0
    merged_meta: dict[str, str] = {}

    for d in datasets:
        activation_chunks.append(d.activations)
        label_chunks.append(d.labels)
        sample_id_chunks.append(d.sample_id + offset)
        offset += int(np.unique(d.sample_id).size)
        n_pos_total += d.n_positive
        n_neg_total += d.n_negative
        merged_meta.update(d.metadata)

    return ActivationDataset(
        activations=np.concatenate(activation_chunks, axis=0),
        labels=np.concatenate(label_chunks, axis=0),
        sample_id=np.concatenate(sample_id_chunks, axis=0),
        scenario=first.scenario,
        layer=first.layer,
        model_name=first.model_name,
        n_positive=n_pos_total,
        n_negative=n_neg_total,
        metadata=merged_meta,
    )


def _sample_level_labels_by_id(
    dataset: ActivationDataset,
) -> tuple[list[int], list[int]]:
    """Return (positive_sample_ids, negative_sample_ids) from a dataset.

    Per-sample label consistency is already enforced by
    ActivationDataset.__post_init__, so the first token's label is
    representative.
    """
    n_samples = int(dataset.sample_id.max() + 1) if dataset.sample_id.size > 0 else 0
    pos_ids: list[int] = []
    neg_ids: list[int] = []
    for sid in range(n_samples):
        mask = dataset.sample_id == sid
        label = int(dataset.labels[mask][0])
        if label == 1:
            pos_ids.append(sid)
        else:
            neg_ids.append(sid)
    return pos_ids, neg_ids


def _evaluate_probe_on_test(
    probe_result,  # type: ignore[no-untyped-def]  -- ProbeResult, avoid circular
    test_dataset: ActivationDataset,
    aggregation: AggregationMethod,
    n_boot: int,
    seed: int,
) -> ProbeMetrics:
    """Apply a trained probe to a test ActivationDataset and compute metrics.

    Mirrors the headline scoring path: predict_probe -> per-token scores ->
    aggregate_token_scores by sample_id -> compute_metrics with bootstrap CIs.
    """
    token_scores = predict_probe(probe_result, test_dataset.activations)
    sample_scores, _ = aggregate_token_scores(
        token_scores, test_dataset.sample_id, method=aggregation
    )

    # Per-sample labels in canonical 0..n-1 order
    n_samples = int(test_dataset.sample_id.max() + 1)
    sample_labels = np.empty(n_samples, dtype=np.int64)
    for sid in range(n_samples):
        sample_labels[sid] = int(test_dataset.labels[test_dataset.sample_id == sid][0])

    return compute_metrics(
        labels=sample_labels,
        scores=sample_scores,
        train_scenario=probe_result.train_scenario,
        eval_scenario=test_dataset.scenario,
        n_boot=n_boot,
        seed=seed,
    )


def _plot_learning_curve(
    n_list: list[int],
    per_n_aurocs: dict[int, list[float]],
    apollo_baseline: float,
    output_path: Path,
    mask: MaskScope,
    aggregation: AggregationMethod,
) -> None:
    """Plot AUROC vs N with mean ± std error bars and the Apollo baseline."""
    means = [float(np.mean(per_n_aurocs[n])) for n in n_list]
    stds = [float(np.std(per_n_aurocs[n], ddof=1)) for n in n_list]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        n_list,
        means,
        yerr=stds,
        marker="o",
        markersize=8,
        capsize=6,
        linewidth=2,
        color="#1f77b4",
        label="Fresh probe (mean ± std across seeds)",
    )

    # Apollo baseline reference line + chance
    ax.axhline(
        apollo_baseline,
        color="#555555",
        linestyle="--",
        linewidth=1.5,
        label=f"Apollo published probe (AUROC {apollo_baseline:.2f})",
    )
    ax.axhline(0.5, color="#888888", linestyle=":", linewidth=1.2, label="chance (0.5)")

    ax.set_xlabel("N (AF training examples)", fontsize=13)
    ax.set_ylabel(f"AUROC ({aggregation} aggregation) ↑ better", fontsize=13)
    ax.set_title(
        f"Stage 2 learning curve: probe AUROC vs N AF examples\n"
        f"(mask={mask}, aggregation={aggregation})",
        fontsize=15,
    )
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(n_list)
    ax.legend(loc="lower right", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Per-N point estimates as text
    for n, mean in zip(n_list, means, strict=True):
        ax.annotate(
            f"{mean:.2f}",
            xy=(n, mean),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--combined-activations",
        type=Path,
        required=True,
        help=(
            "Path to the AF activation .npz containing BOTH positives and "
            "negatives (the file produced by headline_apollo_to_af.py "
            "--save-activations). Stage 2 splits this into train/test "
            "internally."
        ),
    )
    parser.add_argument(
        "--apollo-activations",
        type=Path,
        default=None,
        help=(
            "Optional path to Apollo's roleplaying contrast-pair activations "
            "(.npz from extract_apollo_roleplaying_activations.py). Ignored "
            "under --mask scratchpad (Apollo's data has no scratchpad)."
        ),
    )
    parser.add_argument(
        "--mask",
        choices=_VALID_MASKS,
        default="scratchpad",
        help="Detection mask scope (validated against the .npz metadata).",
    )
    parser.add_argument(
        "--aggregation",
        choices=("mean", "max", "last"),
        default="mean",
        help="Per-token -> per-sample aggregation. Mean matches the headline.",
    )
    parser.add_argument(
        "--n-list",
        type=int,
        nargs="+",
        default=list(_DEFAULT_N_LIST),
        help="N values for the learning curve (number of AF training examples).",
    )
    parser.add_argument(
        "--n-test-positives",
        type=int,
        default=_DEFAULT_N_TEST_POSITIVES,
        help="Hold this many AF positives out as the fixed test set.",
    )
    parser.add_argument(
        "--n-test-negatives",
        type=int,
        default=15,
        help="Hold this many negatives out as the fixed test set.",
    )
    parser.add_argument(
        "--k-seeds",
        type=int,
        default=_DEFAULT_K_SEEDS,
        help="Number of independent seeds per N (controls error-bar tightness).",
    )
    parser.add_argument(
        "--apollo-baseline",
        type=float,
        default=None,
        help=(
            "Override the Apollo published-probe baseline used as the "
            "reference line on the plot. Defaults to a mask-appropriate "
            "value from the headline experiment."
        ),
    )
    parser.add_argument(
        "--master-seed",
        type=int,
        default=42,
        help="Master seed for train/test split and per-seed RNG enumeration.",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=_N_BOOT,
        help="Bootstrap resamples for per-seed AUROC CIs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/learning_curve"),
        help="Where to write learning_curve.json and the curve PNG.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask: MaskScope = args.mask  # type: ignore[assignment]
    aggregation: AggregationMethod = args.aggregation  # type: ignore[assignment]

    print(f"Loading AF activations from {args.combined_activations}")
    af_combined = load_activation_dataset(args.combined_activations)
    validate_scenario(af_combined.scenario)
    expected_mask = af_combined.metadata.get("detection_mask_kind")
    if expected_mask != mask:
        raise ValueError(
            f"--mask {mask!r} does not match the .npz's detection_mask_kind "
            f"{expected_mask!r}. Re-extract with the matching mask or pass "
            f"--mask {expected_mask!r}."
        )

    pos_ids, neg_ids = _sample_level_labels_by_id(af_combined)
    print(
        f"  shape={af_combined.activations.shape}; "
        f"n_positive_samples={len(pos_ids)}, n_negative_samples={len(neg_ids)}"
    )

    if args.n_test_positives >= len(pos_ids):
        raise ValueError(
            f"--n-test-positives ({args.n_test_positives}) >= available "
            f"positives ({len(pos_ids)}); no training pool left."
        )
    if args.n_test_negatives >= len(neg_ids):
        raise ValueError(
            f"--n-test-negatives ({args.n_test_negatives}) >= available "
            f"negatives ({len(neg_ids)}); no training pool left."
        )
    max_n = max(args.n_list)
    train_pool_positives = len(pos_ids) - args.n_test_positives
    if max_n > train_pool_positives:
        raise ValueError(
            f"max(--n-list)={max_n} > training pool positives ({train_pool_positives})"
        )

    rng = np.random.default_rng(args.master_seed)
    pos_ids_shuffled = list(rng.permutation(pos_ids))
    neg_ids_shuffled = list(rng.permutation(neg_ids))
    test_pos_ids = pos_ids_shuffled[: args.n_test_positives]
    train_pool_pos_ids = pos_ids_shuffled[args.n_test_positives :]
    test_neg_ids = neg_ids_shuffled[: args.n_test_negatives]
    train_pool_neg_ids = neg_ids_shuffled[args.n_test_negatives :]

    test_dataset = _select_samples(af_combined, test_pos_ids + test_neg_ids)
    print(
        f"Test set: {test_dataset.n_positive} pos + {test_dataset.n_negative} neg "
        f"(held-out, fixed across all N and seeds)"
    )
    print(f"Training pool: {len(train_pool_pos_ids)} pos + {len(train_pool_neg_ids)} neg available")

    # Optional Apollo training data
    apollo_dataset: ActivationDataset | None = None
    if args.apollo_activations is not None:
        if mask == "scratchpad":
            print(
                "  WARN: --apollo-activations supplied but --mask=scratchpad; "
                "Apollo's prewritten roleplaying data has no scratchpad tokens. "
                "Ignoring the flag (pure-AF training)."
            )
        else:
            apollo_dataset = load_activation_dataset(args.apollo_activations)
            print(
                f"Loaded Apollo training data: shape={apollo_dataset.activations.shape}, "
                f"n_pos={apollo_dataset.n_positive}, n_neg={apollo_dataset.n_negative}"
            )

    apollo_baseline = args.apollo_baseline
    if apollo_baseline is None:
        apollo_baseline = _APOLLO_BASELINE_BY_MASK[mask]
        print(f"Using Apollo-baseline reference: {apollo_baseline:.4f} (mask={mask})")

    # Learning curve
    per_n_aurocs: dict[int, list[float]] = {n: [] for n in args.n_list}
    per_seed_records: list[dict[str, object]] = []

    for n in args.n_list:
        for k in range(args.k_seeds):
            seed = args.master_seed * 1000 + n * 100 + k
            seed_rng = np.random.default_rng(seed)
            sampled_pos_ids = list(seed_rng.choice(train_pool_pos_ids, size=n, replace=False))

            af_train = _select_samples(
                af_combined,
                sampled_pos_ids + list(train_pool_neg_ids),
            )

            if apollo_dataset is not None:
                training_dataset = _combine_datasets([af_train, apollo_dataset])
            else:
                training_dataset = af_train

            probe = train_probe(training_dataset, seed=seed)
            metrics = _evaluate_probe_on_test(probe, test_dataset, aggregation, args.n_boot, seed)

            per_n_aurocs[n].append(metrics.auroc)
            per_seed_records.append(
                {
                    "n": n,
                    "seed": seed,
                    "k": k,
                    "auroc": metrics.auroc,
                    "auroc_ci": list(metrics.auroc_ci) if metrics.auroc_ci else None,
                    "tpr_at_5pct_fpr": metrics.tpr_at_5pct_fpr,
                    "train_auroc_cv": probe.train_auroc,
                    "n_train_samples": probe.n_train,
                }
            )
            print(
                f"  N={n:>3}, k={k:>2}, seed={seed}: AUROC={metrics.auroc:.4f} "
                f"[{metrics.auroc_ci[0]:.3f}, {metrics.auroc_ci[1]:.3f}], "
                f"cv_train_auroc={probe.train_auroc:.3f}"
            )

    summary = {
        "config": {
            "combined_activations": str(args.combined_activations),
            "apollo_activations": (
                str(args.apollo_activations) if args.apollo_activations else None
            ),
            "mask": mask,
            "aggregation": aggregation,
            "n_list": args.n_list,
            "n_test_positives": args.n_test_positives,
            "n_test_negatives": args.n_test_negatives,
            "k_seeds": args.k_seeds,
            "master_seed": args.master_seed,
            "n_boot": args.n_boot,
            "apollo_baseline": apollo_baseline,
            "test_pos_ids": [int(x) for x in test_pos_ids],
            "test_neg_ids": [int(x) for x in test_neg_ids],
        },
        "per_n_summary": {
            str(n): {
                "mean_auroc": float(np.mean(per_n_aurocs[n])),
                "std_auroc": float(np.std(per_n_aurocs[n], ddof=1)),
                "min_auroc": float(np.min(per_n_aurocs[n])),
                "max_auroc": float(np.max(per_n_aurocs[n])),
            }
            for n in args.n_list
        },
        "per_seed_records": per_seed_records,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    json_path = args.output_dir / "learning_curve.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary -> {json_path}")

    plot_path = args.output_dir / "learning_curve.png"
    _plot_learning_curve(
        args.n_list,
        per_n_aurocs,
        apollo_baseline,
        plot_path,
        mask,
        aggregation,
    )

    print("\n=== Stage 2 summary ===")
    for n in args.n_list:
        mean = float(np.mean(per_n_aurocs[n]))
        std = float(np.std(per_n_aurocs[n], ddof=1))
        print(f"  N={n:>3}: mean AUROC = {mean:.4f} ± {std:.4f} (n_seeds={args.k_seeds})")
    print(f"  Apollo baseline: {apollo_baseline:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
