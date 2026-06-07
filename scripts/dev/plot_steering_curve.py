"""Plot the alpha-sweep steering curve for Blog 4's causal-intervention section.

Loads ``data/processed/steering_repe_classified.jsonl`` (output of
``scripts/dev/classify_steered_outputs.py``) and produces two figures
that slot into Blog 4's mechanism section alongside Figure 4:

1. ``docs/figures/fig6_steering_curve.png`` -- bidirectional alpha sweep:
   fraction of transcripts classified as canonical AF (confidence >= 0.6)
   as a function of steering strength alpha, with Wilson 95% CIs.
   Two lines: af_partial cohort (full alpha range) and af-positive
   control cohort (negative alpha only).

2. ``docs/figures/fig7_steering_examples.png`` -- decoded examples panel:
   2-3 representative af_partial transcripts at alpha=0, +1, +2 with
   the response text excerpted, so a reader can see the qualitative
   shift the curve summarises.

Pure-CPU, local script. No GPU needed.

Styling matches ``scripts/dev/make_presentation_figures.py``:
16:9 aspect at 200 DPI, slide-friendly font sizes, project colour palette.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Slide-friendly styling -- matches make_presentation_figures.py
FIGURE_WIDTH_IN = 12.0
FIGURE_HEIGHT_IN = 6.75
DPI = 200
TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 14
TICK_FONTSIZE = 12
VALUE_FONTSIZE = 13
LEGEND_FONTSIZE = 12

# Colour palette: keep RePE = purple to match cross-probe figure
COLOR_AF_PARTIAL = "#d62728"  # red: af_partial cohort
COLOR_AF_POSITIVE = "#2ca02c"  # green: af-positive control cohort
COLOR_CHANCE = "#888888"  # grey: reference / chance lines
COLOR_BASELINE_PARTIAL = "#a04040"  # darker red: af_partial baseline marker
COLOR_BASELINE_AF = "#1a6e1a"  # darker green: af baseline marker

# Classification threshold for "canonical AF" (matches af_classifier.py
# convention: 5-criteria classifier returns confidence in [0, 1] equal
# to the fraction of criteria scored "yes"; >= 0.6 means >= 3/5).
AF_CONFIDENCE_THRESHOLD = 0.6


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI for a binomial proportion.

    Better than Wald at small n and at proportions near 0 or 1, which is
    exactly the regime we land in (n=20 af_partial, n=5 af control).
    Returns (lo, hi) -- the point estimate is k/n.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo = max(0.0, centre - half)
    hi = min(1.0, centre + half)
    return lo, hi


def _load_classified(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _summarise_by_alpha(
    rows: list[dict[str, Any]],
    cohort: str,
) -> list[tuple[float, int, int, float, float]]:
    """Aggregate (alpha -> fraction classified as canonical AF) for one cohort.

    Returns a list of ``(alpha, k_positive, n_total, ci_lo, ci_hi)`` rows
    sorted by alpha. A row is counted as canonical-AF if
    ``label == 1 AND classifier_confidence >= AF_CONFIDENCE_THRESHOLD``.
    """
    by_alpha: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["cohort"] != cohort:
            continue
        by_alpha[float(row["alpha"])].append(row)

    summary: list[tuple[float, int, int, float, float]] = []
    for alpha in sorted(by_alpha.keys()):
        items = by_alpha[alpha]
        n = len(items)
        k = sum(
            1
            for r in items
            if int(r["label"]) == 1 and float(r["classifier_confidence"]) >= AF_CONFIDENCE_THRESHOLD
        )
        lo, hi = _wilson_ci(k, n)
        summary.append((alpha, k, n, lo, hi))
    return summary


def make_figure_steering_curve(
    classified_path: Path,
    output_path: Path,
) -> None:
    """Bidirectional alpha-sweep curve.

    X: steering strength alpha (probe-direction-norm units).
    Y: fraction of transcripts classified as canonical AF.
    Two lines, Wilson 95% CIs as error bars, alpha=0 highlighted.
    """
    rows = _load_classified(classified_path)
    partial = _summarise_by_alpha(rows, cohort="af_partial")
    control = _summarise_by_alpha(rows, cohort="af_control")

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), dpi=DPI)

    # --- af_partial line (full alpha sweep, primary cohort)
    if partial:
        alphas_p = np.array([r[0] for r in partial])
        fracs_p = np.array([r[1] / r[2] if r[2] else 0.0 for r in partial])
        los_p = np.array([r[3] for r in partial])
        his_p = np.array([r[4] for r in partial])
        yerr_p = np.vstack([fracs_p - los_p, his_p - fracs_p])
        n_partial = partial[0][2]
        ax.errorbar(
            alphas_p,
            fracs_p,
            yerr=yerr_p,
            marker="o",
            markersize=8,
            linewidth=2,
            capsize=4,
            color=COLOR_AF_PARTIAL,
            label=f"af_partial cohort (n={n_partial})",
        )

    # --- af-positive control line (negative alpha + 0, flip-to-refusal control)
    if control:
        alphas_c = np.array([r[0] for r in control])
        fracs_c = np.array([r[1] / r[2] if r[2] else 0.0 for r in control])
        los_c = np.array([r[3] for r in control])
        his_c = np.array([r[4] for r in control])
        yerr_c = np.vstack([fracs_c - los_c, his_c - fracs_c])
        n_control = control[0][2]
        ax.errorbar(
            alphas_c,
            fracs_c,
            yerr=yerr_c,
            marker="s",
            markersize=8,
            linewidth=2,
            capsize=4,
            color=COLOR_AF_POSITIVE,
            label=f"af-positive control (n={n_control})",
        )

    # alpha=0 vertical reference (the "steering off" line)
    ax.axvline(0.0, color=COLOR_CHANCE, linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(
        0.05,
        0.02,
        "alpha = 0\n(steering off)",
        transform=ax.get_xaxis_transform(),
        fontsize=TICK_FONTSIZE - 1,
        color=COLOR_CHANCE,
        verticalalignment="bottom",
    )

    ax.set_xlabel(
        "Steering strength alpha (in probe-direction-norm units)",
        fontsize=LABEL_FONTSIZE,
    )
    ax.set_ylabel(
        "Fraction classified as canonical AF\n(classifier confidence >= 0.6)",
        fontsize=LABEL_FONTSIZE,
    )
    ax.set_title(
        "Steering along the RePE probe direction at layer 22",
        fontsize=TITLE_FONTSIZE,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.legend(loc="upper left", fontsize=LEGEND_FONTSIZE)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def _example_row_for(
    outputs: list[dict[str, Any]],
    transcript_idx: int,
    cohort: str,
    alpha: float,
) -> dict[str, Any] | None:
    for row in outputs:
        if (
            row.get("cohort") == cohort
            and int(row.get("transcript_idx", -1)) == transcript_idx
            and math.isclose(float(row.get("alpha", math.nan)), alpha, abs_tol=1e-6)
        ):
            return row
    return None


def _truncate(text: str | None, max_chars: int = 350) -> str:
    if not text:
        return "(no text)"
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


def make_figure_steering_examples(
    outputs_path: Path,
    output_path: Path,
    *,
    transcript_indices: tuple[int, ...] = (0, 1, 2),
    alphas: tuple[float, ...] = (0.0, 1.0, 2.0),
) -> None:
    """Decoded-examples panel: a few representative af_partial transcripts
    shown at alpha = 0 vs alpha = +1 vs alpha = +2.

    Pulls raw text from the unclassified outputs JSONL (not the classified
    one) so the actual response strings are available. Skips silently if
    the outputs file is missing -- this figure is optional polish.
    """
    if not outputs_path.exists():
        print(f"Skipping examples figure: {outputs_path} not found.")
        return

    outputs: list[dict[str, Any]] = []
    with outputs_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            outputs.append(json.loads(line))

    n_rows = len(transcript_indices)
    n_cols = len(alphas)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(FIGURE_WIDTH_IN * 1.2, FIGURE_HEIGHT_IN * n_rows / 2.0),
        dpi=DPI,
    )
    if n_rows == 1:
        axes = np.array([axes])

    for r, t_idx in enumerate(transcript_indices):
        for c, alpha in enumerate(alphas):
            ax = axes[r, c]
            ax.axis("off")
            row = _example_row_for(outputs, t_idx, "af_partial", alpha)
            if row is None:
                ax.text(
                    0.5,
                    0.5,
                    f"(missing)\ntranscript {t_idx}, alpha={alpha:+g}",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=TICK_FONTSIZE,
                    color=COLOR_CHANCE,
                )
                continue
            title = f"transcript {t_idx} | alpha={alpha:+g}"
            response_text = _truncate(row.get("response"))
            ax.text(
                0.02,
                0.98,
                title,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=TICK_FONTSIZE,
                fontweight="bold",
            )
            ax.text(
                0.02,
                0.88,
                response_text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=TICK_FONTSIZE - 2,
                wrap=True,
                family="monospace",
            )
            # Light border
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(COLOR_CHANCE)
                spine.set_linewidth(0.5)

    fig.suptitle(
        "Decoded examples: alpha-sweep on af_partial transcripts",
        fontsize=TITLE_FONTSIZE,
        y=1.0,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classified",
        type=Path,
        default=Path("data/processed/steering_repe_classified.jsonl"),
        help="JSONL of per-row classifier labels.",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("data/processed/steering_repe_outputs.jsonl"),
        help=(
            "JSONL of raw steered generations (for the examples figure). "
            "Optional -- examples figure is skipped if missing."
        ),
    )
    parser.add_argument(
        "--curve-output",
        type=Path,
        default=Path("docs/figures/fig6_steering_curve.png"),
    )
    parser.add_argument(
        "--examples-output",
        type=Path,
        default=Path("docs/figures/fig7_steering_examples.png"),
    )
    parser.add_argument(
        "--example-transcripts",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Transcript indices to show in the examples panel.",
    )
    parser.add_argument(
        "--example-alphas",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 2.0],
        help="Alpha values to show in the examples panel.",
    )
    parser.add_argument(
        "--skip-examples",
        action="store_true",
        help="Only generate the curve figure.",
    )
    args = parser.parse_args()

    if not args.classified.exists():
        raise FileNotFoundError(
            f"Classified file not found: {args.classified}. "
            f"Run scripts/dev/classify_steered_outputs.py first."
        )

    make_figure_steering_curve(args.classified, args.curve_output)

    if not args.skip_examples:
        make_figure_steering_examples(
            args.outputs,
            args.examples_output,
            transcript_indices=tuple(args.example_transcripts),
            alphas=tuple(args.example_alphas),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
