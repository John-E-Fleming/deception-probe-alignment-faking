"""Generate slide-ready figures for the mentor presentation.

Self-contained: all experimental results are hardcoded as constants so
the script reproduces the exact numbers in the docs without re-running
extractions or loading .npz files.

Produces three PNGs at slide-friendly resolution (16:9 aspect, 200 DPI)
under ``docs/figures/``:

- ``fig1_headline_auroc.png``: bar chart, AUROC across the 3 mask
  scopes (refusal negatives, mean aggregation) with the in-distribution
  anchor as a dashed reference line. The "experiments worked" headline.
- ``fig2_localisation.png``: two-panel chart comparing aggregation
  methods on response-only vs scratchpad-only. Shows that max-pool
  rescues a localised response signal but scratchpad is at chance under
  every aggregation.
- ``fig3_sensitivity_test.png``: grouped bar chart, 3 mask scopes × 2
  negative classes (refusal vs af_partial), mean aggregation. The
  "but actually" figure -- signal collapses when negatives carry
  compliance content.

Run locally; no GPU or pod dependencies. Outputs land in docs/figures/
and are git-committable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Experimental results (locked 2026-06-02 across the headline run +
# ablations + sensitivity test). Values are AUROC point estimate and
# 95% stratified-bootstrap CI (n_boot = 10000).

# Refusal negatives: 19 af positives + 50 free-tier clean refusals.
RESULTS_REFUSAL: dict[str, dict[str, tuple[float, float, float]]] = {
    "scratchpad": {
        "mean": (0.6032, 0.4347, 0.7684),
        "max": (0.5063, 0.3189, 0.6926),
        "last": (0.3000, 0.1610, 0.4579),
    },
    "response": {
        "mean": (0.7116, 0.5811, 0.8253),
        "max": (0.9800, 0.9463, 1.0000),
        "last": (0.7505, 0.6179, 0.8684),
    },
    "both": {
        "mean": (0.9221, 0.8484, 0.9779),
        "max": (0.9589, 0.9095, 0.9937),
        "last": (0.7505, 0.6179, 0.8684),
    },
}

# af_partial negatives: 19 af positives + 20 af_partial (softer-negative
# sensitivity test).
RESULTS_AF_PARTIAL: dict[str, dict[str, tuple[float, float, float]]] = {
    "scratchpad": {
        "mean": (0.3763, 0.2053, 0.5632),
        "max": (0.3000, 0.1474, 0.4763),
        "last": (0.3237, 0.1658, 0.5026),
    },
    "response": {
        "mean": (0.5289, 0.3368, 0.7105),
        "max": (0.6658, 0.4842, 0.8263),
        "last": (0.6421, 0.4500, 0.8211),
    },
    "both": {
        "mean": (0.5500, 0.3632, 0.7395),
        "max": (0.5289, 0.3474, 0.7132),
        "last": (0.6421, 0.4500, 0.8211),
    },
}

# Apollo's roleplaying probe on roleplaying__offpolicy_train with our
# adapter + pipeline. Response-only detection, mean aggregation. No CI
# reported by check_apollo_format (it doesn't run bootstrap by default).
ANCHOR_AUROC = 0.8190

# Slide-friendly styling
FIGURE_WIDTH_IN = 12.0
FIGURE_HEIGHT_IN = 6.75  # 16:9 ratio
DPI = 200
TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 14
TICK_FONTSIZE = 12
VALUE_FONTSIZE = 13
LEGEND_FONTSIZE = 12

# Color palette: muted but high-contrast for slide projection
COLOR_REFUSAL = "#1f77b4"
COLOR_AF_PARTIAL = "#d62728"
COLOR_ANCHOR = "#555555"
COLOR_CHANCE = "#888888"


def _err_from_ci(point: float, lo: float, hi: float) -> tuple[float, float]:
    """Convert (point, lo, hi) into the (lower, upper) error-bar offsets
    matplotlib expects (asymmetric)."""
    return (point - lo, hi - point)


def _annotate_bar(ax: plt.Axes, x: float, height: float, label: str) -> None:
    """Put a value label above the bar (and above its error-bar)."""
    ax.text(
        x,
        height + 0.04,
        label,
        ha="center",
        va="bottom",
        fontsize=VALUE_FONTSIZE,
        fontweight="bold",
    )


def make_figure_1_headline(output_path: Path) -> None:
    """Fig 1: AUROC across mask scopes (refusal negatives, mean aggregation)
    with the in-distribution anchor as reference."""
    mask_scopes = ["scratchpad", "response", "both"]
    points = [RESULTS_REFUSAL[m]["mean"][0] for m in mask_scopes]
    lows = [RESULTS_REFUSAL[m]["mean"][1] for m in mask_scopes]
    highs = [RESULTS_REFUSAL[m]["mean"][2] for m in mask_scopes]

    yerr_lower = [p - lo for p, lo in zip(points, lows, strict=True)]
    yerr_upper = [hi - p for p, hi in zip(points, highs, strict=True)]

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN))

    x = np.arange(len(mask_scopes))
    ax.bar(
        x,
        points,
        yerr=[yerr_lower, yerr_upper],
        color=COLOR_REFUSAL,
        edgecolor="black",
        linewidth=0.8,
        capsize=6,
        width=0.55,
    )

    # Chance line + anchor line
    ax.axhline(0.5, color=COLOR_CHANCE, linestyle=":", linewidth=1.2, zorder=1)
    ax.text(
        len(mask_scopes) - 0.6,
        0.52,
        "chance (0.5)",
        fontsize=11,
        color=COLOR_CHANCE,
        va="bottom",
    )
    ax.axhline(
        ANCHOR_AUROC,
        color=COLOR_ANCHOR,
        linestyle="--",
        linewidth=1.5,
        zorder=1,
    )
    ax.text(
        len(mask_scopes) - 0.6,
        ANCHOR_AUROC + 0.015,
        f"in-distribution anchor ({ANCHOR_AUROC:.2f})",
        fontsize=11,
        color=COLOR_ANCHOR,
        va="bottom",
        style="italic",
    )

    # Value labels on bars
    for xi, p, hi in zip(x, points, highs, strict=True):
        _annotate_bar(ax, xi, hi, f"{p:.2f}")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.capitalize() for m in mask_scopes],
        fontsize=LABEL_FONTSIZE,
    )
    ax.set_xlabel("Detection mask scope", fontsize=LABEL_FONTSIZE, labelpad=10)
    ax.set_ylabel(
        "AUROC (mean aggregation)  ↑ better",
        fontsize=LABEL_FONTSIZE,
        labelpad=10,
    )
    ax.set_ylim(0.0, 1.1)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax.set_title(
        "Apollo's roleplaying probe → naturally-emerging AF transcripts\n"
        "n_positive=19 af, n_negative=50 free-tier refusals, 95% bootstrap CIs",
        fontsize=TITLE_FONTSIZE,
        pad=14,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {output_path}")


def make_figure_2_localisation(output_path: Path) -> None:
    """Fig 2: two-panel comparison of aggregation methods on response-only
    vs scratchpad-only. Shows max-pool rescues localised response signal
    but scratchpad is at chance under every aggregation."""
    aggregations = ["mean", "max", "last"]

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN),
        sharey=True,
    )

    for ax, mask, panel_title in [
        (ax_left, "response", "Response tokens only"),
        (ax_right, "scratchpad", "Scratchpad tokens only"),
    ]:
        points = [RESULTS_REFUSAL[mask][a][0] for a in aggregations]
        lows = [RESULTS_REFUSAL[mask][a][1] for a in aggregations]
        highs = [RESULTS_REFUSAL[mask][a][2] for a in aggregations]
        yerr_lower = [p - lo for p, lo in zip(points, lows, strict=True)]
        yerr_upper = [hi - p for p, hi in zip(points, highs, strict=True)]

        x = np.arange(len(aggregations))
        ax.bar(
            x,
            points,
            yerr=[yerr_lower, yerr_upper],
            color=COLOR_REFUSAL,
            edgecolor="black",
            linewidth=0.8,
            capsize=6,
            width=0.55,
        )

        # Chance line
        ax.axhline(0.5, color=COLOR_CHANCE, linestyle=":", linewidth=1.2, zorder=1)

        # Value labels on bars
        for xi, p, hi in zip(x, points, highs, strict=True):
            _annotate_bar(ax, xi, hi, f"{p:.2f}")

        ax.set_xticks(x)
        ax.set_xticklabels(
            [a.capitalize() for a in aggregations],
            fontsize=LABEL_FONTSIZE,
        )
        ax.set_xlabel("Per-token aggregation", fontsize=LABEL_FONTSIZE, labelpad=10)
        ax.set_title(panel_title, fontsize=LABEL_FONTSIZE + 1, pad=8)
        ax.set_ylim(0.0, 1.1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)

    ax_left.set_ylabel("AUROC  ↑ better", fontsize=LABEL_FONTSIZE, labelpad=10)
    # Annotate "chance" on the left panel only (cleaner)
    ax_left.text(
        len(aggregations) - 0.5,
        0.52,
        "chance",
        fontsize=11,
        color=COLOR_CHANCE,
        va="bottom",
    )

    fig.suptitle(
        "Response signal is localised; scratchpad signal is absent\n"
        "(refusal negatives, n_pos=19, n_neg=50)",
        fontsize=TITLE_FONTSIZE,
        y=1.02,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {output_path}")


def make_figure_3_sensitivity(output_path: Path) -> None:
    """Fig 3: grouped bar chart, 3 mask scopes × 2 negative classes
    (refusal vs af_partial), mean aggregation. The 'killer' figure
    showing the signal collapses when negatives carry compliance content."""
    mask_scopes = ["scratchpad", "response", "both"]

    refusal_points = [RESULTS_REFUSAL[m]["mean"][0] for m in mask_scopes]
    refusal_lows = [RESULTS_REFUSAL[m]["mean"][1] for m in mask_scopes]
    refusal_highs = [RESULTS_REFUSAL[m]["mean"][2] for m in mask_scopes]

    afp_points = [RESULTS_AF_PARTIAL[m]["mean"][0] for m in mask_scopes]
    afp_lows = [RESULTS_AF_PARTIAL[m]["mean"][1] for m in mask_scopes]
    afp_highs = [RESULTS_AF_PARTIAL[m]["mean"][2] for m in mask_scopes]

    refusal_yerr = (
        [p - lo for p, lo in zip(refusal_points, refusal_lows, strict=True)],
        [hi - p for p, hi in zip(refusal_points, refusal_highs, strict=True)],
    )
    afp_yerr = (
        [p - lo for p, lo in zip(afp_points, afp_lows, strict=True)],
        [hi - p for p, hi in zip(afp_points, afp_highs, strict=True)],
    )

    x = np.arange(len(mask_scopes))
    bar_width = 0.36

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN))

    ax.bar(
        x - bar_width / 2,
        refusal_points,
        bar_width,
        yerr=refusal_yerr,
        color=COLOR_REFUSAL,
        edgecolor="black",
        linewidth=0.8,
        capsize=5,
        label="Refusal negatives (n=50)",
    )
    ax.bar(
        x + bar_width / 2,
        afp_points,
        bar_width,
        yerr=afp_yerr,
        color=COLOR_AF_PARTIAL,
        edgecolor="black",
        linewidth=0.8,
        capsize=5,
        label="af_partial negatives (n=20)",
    )

    # Chance line
    ax.axhline(0.5, color=COLOR_CHANCE, linestyle=":", linewidth=1.2, zorder=1)
    ax.text(
        -0.45,
        0.52,
        "chance",
        fontsize=11,
        color=COLOR_CHANCE,
        va="bottom",
    )

    # Value labels on bars
    for xi, p, hi in zip(x - bar_width / 2, refusal_points, refusal_highs, strict=True):
        _annotate_bar(ax, xi, hi, f"{p:.2f}")
    for xi, p, hi in zip(x + bar_width / 2, afp_points, afp_highs, strict=True):
        _annotate_bar(ax, xi, hi, f"{p:.2f}")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.capitalize() for m in mask_scopes],
        fontsize=LABEL_FONTSIZE,
    )
    ax.set_xlabel("Detection mask scope", fontsize=LABEL_FONTSIZE, labelpad=10)
    ax.set_ylabel(
        "AUROC (mean aggregation)  ↑ better",
        fontsize=LABEL_FONTSIZE,
        labelpad=10,
    )
    ax.set_ylim(0.0, 1.15)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax.set_title(
        "Sensitivity test: signal collapses when negatives carry compliance content\n"
        "(n_positive=19 af; 95% bootstrap CIs)",
        fontsize=TITLE_FONTSIZE,
        pad=14,
    )
    ax.legend(
        loc="upper left",
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        edgecolor="black",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/figures"),
        help="Where to write the slide-ready PNGs.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing slide figures to {args.output_dir}")

    make_figure_1_headline(args.output_dir / "fig1_headline_auroc.png")
    make_figure_2_localisation(args.output_dir / "fig2_localisation.png")
    make_figure_3_sensitivity(args.output_dir / "fig3_sensitivity_test.png")

    print()
    print("Note: Figure 4 (per-token decoded examples) is not auto-generated.")
    print("It needs human curation of which transcripts to show, AND should")
    print("wait until the activations<->JSONL alignment bug is resolved so")
    print("the AdvBench prompts are paired with the right response content.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
