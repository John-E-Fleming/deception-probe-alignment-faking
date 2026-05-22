# deception-probe-alignment-faking

Testing whether linear probes trained on instructed deception generalise to
naturally-emerging alignment faking. Continues an ongoing research series on
probe-based deception detection in LLMs.

> **Status:** Week 1 of a 4-week sprint — repo setup complete, no experiments
> run yet. Interface, results, and documentation are subject to change.

---

## Research question

Apollo Research's deception probes
([Goldowsky-Dill et al., 2025](https://arxiv.org/abs/2502.03407)) achieve
0.96–0.999 AUROC on three *instructed or incentivised* deception scenarios:
roleplaying, insider trading, and sandbagging. Apollo themselves note these
probes are "insufficient as a robust defence against deception" and do not
test alignment faking.

[Kirch et al. (2025)](https://arxiv.org/abs/2511.17408) subsequently showed
that off-policy training data causes systematic generalisation failures for
intent-based behaviours like deception — probes latch onto surface features of
the *training* deception rather than the underlying intent.

Alignment faking
([Greenblatt et al., 2024](https://arxiv.org/abs/2412.14093)) is the hardest
case of this failure mode: the model was never told to deceive, it reasoned
its way there from an instrumental goal.

**The question this repo measures:** do Apollo's probes generalise to
naturally-emerging alignment faking? Either outcome is a contribution. The
experiment is framed as a measurement, not a test with a preferred answer.

---

## The 2×2 transfer matrix

The primary result is a cross-scenario probe transfer matrix on
`meta-llama/Llama-3.3-70B-Instruct`:

|                        | Eval: Apollo scenarios | Eval: Alignment faking |
|------------------------|------------------------|------------------------|
| **Train: Apollo**      | Baseline (≈0.96–0.999) | **Core novel result**  |
| **Train: AF**          | Reverse transfer       | Within-distribution    |

Per cell: AUROC, TPR@1%FPR, TPR@5%FPR. All four cells are required for the
result to be interpretable — the diagonal provides the baselines that frame
the off-diagonal interpretation.

---

## Repository structure

```
alignment_faking_probes/
├── data/                 # Transcript generation, AF labeling, activation extraction, ActivationDataset
├── probes/               # ProbeResult, train_probe, predict_probe, evaluation metrics, plotting
└── transfer/             # TransferCell, run_transfer_matrix — the core experiment

scripts/                  # Numbered pipeline (01 through 06, run in order)
configs/                  # Per-experiment YAML configs (no hardcoded hyperparameters)
tests/                    # Pytest suite (mock-tensor based; @pytest.mark.gpu tests run on RunPod)
notebooks/                # Replication checks, interpretation, figure prototyping
results/                  # Committed final outputs (transfer matrix JSON, ROC curves, metadata)
docs/                     # External-facing project context
```

---

## Setup

### Local development (Windows / Mac / Linux, CPU)

For module development, tests, lint, and typecheck — no GPU required.

```bash
uv sync
uv run pytest tests/ -v
```

`uv` is the only required tool. The project uses Python 3.11+ (see
`.python-version`).

### RunPod (for GPU experiments)

Activation extraction and transcript generation require a GPU with enough
VRAM to load Llama-3.3-70B-Instruct under 4-bit quantisation (~40 GB).
Tested on A100 80GB.

```bash
# 1. Clone this repo and install project deps
git clone https://github.com/John-E-Fleming/deception-probe-alignment-faking
cd deception-probe-alignment-faking
uv sync

# 2. Install Apollo's deception-detection separately (not uv-managed; see
#    pyproject.toml for why). Pinned commit must match across runs.
cd /workspace
git clone https://github.com/ApolloResearch/deception-detection
cd deception-detection
git checkout f8ec4010e74927394709dffa22b97bdf8cd5a62f
pip install -e .
export PYTHONPATH="/workspace/deception-detection:$PYTHONPATH"

# 3. Authenticate with HuggingFace (Llama-3.3 is gated) and W&B
huggingface-cli login    # uses HF_TOKEN from .env
wandb login              # uses WANDB_API_KEY from .env
```

The pinned Apollo commit hash must be logged as a W&B config field on every
experiment run and written into `results/experiment_metadata.json`. See
`pyproject.toml` for the source-of-truth pin.

---

## Pipeline

Scripts are numbered to be run in order. Each writes to `data/`, `results/`,
or both.

| Script | Purpose |
|--------|---------|
| `01_generate_af_transcripts.py` | Generate ~800–1000 alignment faking transcripts from Llama-3.3-70B-Instruct using Redwood's prompt setup |
| `02_label_af_transcripts.py`    | Run Redwood's automated classifier; filter by confidence threshold |
| `03_create_train_test_split.py` | 80/20 stratified split — freezes the AF test set for the rest of the project |
| `04_extract_activations.py`     | Extract residual stream activations for Apollo scenarios and the AF train set |
| `05_run_transfer_experiment.py` | Populate all four cells of the 2×2 transfer matrix; write results to `results/` |
| `06_generate_plots.py`          | ROC curves, transfer matrix heatmap, layer sweep heatmap |

Each script reads its configuration from `configs/<script_name>.yaml`. No
hyperparameters are hardcoded.

---

## Series context

Earlier work in this research series:

- **Linear probes for unfaithful CoT detection**
  ([repo](https://github.com/John-E-Fleming/CoT_Unfaithfulness_Detection_with_Linear_Probes))
  — Probes detect unfaithful chain-of-thought reasoning (AP = 0.484,
  3.8× baseline) on Qwen2.5-7B. Established the core probe training
  infrastructure.
- **Probes under distribution shift**
  ([repo](https://github.com/John-E-Fleming/unfaithfulness-control-monitor))
  — Naive probes collapse out-of-distribution (AUROC 0.754 → 0.590), but
  LOCO training partially recovers performance (→ 0.806). This
  distribution-shift finding directly motivates the present experiment.

This repo extends that line of work to the hardest distribution shift: from
*instructed* to *naturally-emerging* deception. It is intentionally narrow:
one model, one measurement, two scenarios.

---

## Key papers

- **Apollo Research, 2025** —
  [Detecting Strategic Deception Using Linear Probes](https://arxiv.org/abs/2502.03407)
  (Goldowsky-Dill, Chughtai, Heimersheim, Hobbhahn). Replication target.
- **Anthropic / Redwood, 2024** —
  [Alignment Faking in Large Language Models](https://arxiv.org/abs/2412.14093)
  (Greenblatt, Denison, Wright et al.). Source of AF prompt setup and the
  classifier used for labeling.
- **King's College / LASR / Cambridge, 2025** —
  [The Impact of Off-Policy Training Data on Probe Generalisation](https://arxiv.org/abs/2511.17408)
  (Kirch, Dower, Skapars, Yannakoudakis, Lubana, Krasheninnikov). Motivates
  the prediction that off-policy probes will not generalise to alignment
  faking.

See [`docs/bluedot_project_outline.md`](docs/bluedot_project_outline.md) for
the full project framing and the three extension directions beyond this
sprint.

---

## Acknowledgments

Developed during the BlueDot Impact AI Safety sprint cohort.
Built on Apollo Research's `deception-detection` and Redwood Research's
`alignment_faking_public`.
