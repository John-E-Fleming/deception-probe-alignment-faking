# deception-probe-alignment-faking

Testing whether linear probes trained on instructed deception generalise to
naturally-emerging alignment faking. Continues an ongoing research series on
probe-based deception detection in LLMs.

> **Status:** Sprint week 3. **Preliminary headline result (2026-06-01):**
> Apollo's published roleplaying probe applied to 19 behaviourally-confirmed
> AF transcripts + 50 clean refusals on Llama-3.3-70B + Anthropic's
> released LoRA adapter (`jplhughes2/llama-3.3-70b-af-synthetic-docs-only-more-data-r-epochs`)
> at layer 22 yields **AUROC = 0.9221, 95% bootstrap CI [0.8484, 0.9779]**
> (TPR@5%FPR = 0.63 [0.05, 0.89]; n_positive = 19, n_negative = 50,
> n_boot = 10000). The headline sits well above the pre-registered
> "meaningful transfer" threshold (≥ 0.7 with lower CI > 0.5).
>
> **This number is preliminary and conditional on mask-scope ablations
> not yet run.** The headline used `mask_kind="both"` (scoring scratchpad
> + response tokens), so the probe may be detecting compliance-vs-refusal
> at the response level rather than alignment-faking reasoning in the
> scratchpad. The discriminating test — re-aggregating the saved per-token
> activations restricted to scratchpad positions — is the immediate next
> step, no additional GPU required. We are also running an
> in-distribution anchor (Apollo's probe on Apollo's roleplaying
> prewritten data with our adapter) so the headline number has a
> same-pipeline baseline to compare against rather than relying on
> Apollo's published figure. 12-item confound list + ordered next-
> experiment plan in [`docs/findings_log.md`](docs/findings_log.md) under
> the 2026-06-01 entry.
>
> **Pipeline state.** Apollo-backed extraction pipeline verified faithful
> (Spearman = 0.9999, Test #1 2026-05-28). Apollo's published 70B probe
> transfers cleanly to LoRA-modified activations (Spearman = 0.9997,
> AUROC preserved 0.61 → 0.68 on `ai_liar` prewritten). AF generation
> expanded to n=1040 transcripts; classifier under-reports AF, so the
> positive class is set by **manual behavioural inspection** of 45
> candidate transcripts: 19 `af`, 20 `af_partial`, 4 `not_af`, 9
> `artifact`. Classifier calibration against
> `jplhughes2/classify_alignment_faking_human_labels` found Sonnet 4.6
> judge unreliable in both directions (agreement = 0.53), reinforcing
> behavioural inspection as the gold standard.
>
> **Framing (2026-06-01): per-transcript probe detection, not
> population-level compliance gap.** A deployed probe operates per
> transcript at inference time, so the positive-class definition is
> behaviourally-confirmed AF transcripts regardless of pair structure.
> The compliance gap at n=520 per tier is -2.5%pp ± 3.08, 95% CI
> [-8.54, +3.54] (paired-difference SE per Miller 2024 §4.2) —
> statistically indistinguishable from zero. The pair-structure metric
> systematically under-counts real AF: a top-by-confidence inspection
> found 40–80% yield of confirmed `af` cases that sit *outside* textbook-
> AF pairs because per-tier compliance probabilities are nearly
> independent (correlation = +0.008). This is itself a defensible
> methodological finding for the blog post. Adopted Miller (2024)
> statistical best practices for all uncertainty reporting — see
> [`docs/stats_best_practices.md`](docs/stats_best_practices.md).
>
> **Next:** mask-scope and aggregation ablations on the saved
> activations (no GPU, ~30 min), `af_partial` sensitivity check,
> per-token interpretability pass, bootstrap leave-one-out, and a
> single-A100 anchor run (Apollo's probe on `roleplaying__plain` with
> our adapter, ~15 min, ~$0.50). Stage 2 (learning curve with N AF
> examples added to the probe's training set, testing Kirch et al. 2025)
> is on hold until the ablations resolve which reading of the headline
> holds. Full chronology in
> [`docs/findings_log.md`](docs/findings_log.md). Interface, results,
> and documentation subject to change.

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
├── data/                 # Transcript generation, AF generation prompts, AF classifier, labeling,
│                         # per-token activation extraction (Apollo wrapper), ActivationDataset
├── probes/               # ProbeResult, train_probe, predict_probe, aggregate_token_scores,
│                         # evaluation metrics, plotting
└── transfer/             # TransferCell, run_transfer_matrix — the core experiment

data/redwood_honly/       # Vendored from Redwood's alignment_faking_public (MIT) — system prompt
                          # and AF classifier few-shot examples
scripts/                  # Numbered pipeline (01 through 06, run in order) + headline cell
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

Activation extraction and transcript generation require a GPU. Tested on
**A100 80GB** with a **200 GB persistent volume** mounted at `/workspace`.
Recommended template: PyTorch 2.4+ on Ubuntu 22.04 with CUDA driver ≥ 12.6.

**One-time pod setup:**

```bash
# Install uv (RunPod templates don't ship it)
curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL

# Park HF cache on the persistent volume so pod restarts don't trigger a
# 140 GB Llama re-download
echo 'export HF_HOME=/workspace/.cache/huggingface' >> ~/.bashrc
export HF_HOME=/workspace/.cache/huggingface

# Clone this repo and create .env with the required tokens
cd /workspace
git clone https://github.com/John-E-Fleming/deception-probe-alignment-faking
cd deception-probe-alignment-faking
cat > .env <<'EOF'
HF_TOKEN=hf_xxx
WANDB_API_KEY=xxx
ANTHROPIC_API_KEY=sk-ant-xxx   # needed by scripts/02_label_af_transcripts.py
EOF

# Load .env into the current shell, then run the setup script
set -a && source .env && set +a
bash scripts/00_runpod_setup.sh
```

**After every `git pull` on the pod, run the setup script again — not just
`uv sync`.** The setup script does the sync AND reinstalls Apollo's editable
package idempotently. Apollo itself is NOT in `pyproject.toml` (its
`torch<2.3` pin would fight our `torch>=2.4`) — but Apollo's *transitive*
deps (peft, jaxtyping, einops, anthropic, openai, inspect-ai, sae-lens,
together, goodfire, etc.) ARE listed as Linux-only deps so `uv sync` keeps
them across reinstalls. First sync on a fresh pod takes ~5–10 min for
those transitive packages alone. The "post-pull" command is:

```bash
cd /workspace/deception-probe-alignment-faking
git pull
bash scripts/00_runpod_setup.sh
```

**Data persistence across pod restarts.** Runtime data under `data/` (raw
transcripts, labelled transcripts, activation datasets, trained probes) is
gitignored and persisted to a private HuggingFace Hub dataset repo —
`John-E-Fleming/deception-probe-alignment-faking-data`. Once per pod
session:

```bash
make hf-login       # authenticate using HF_TOKEN from .env
make hf-pull-data   # restore any previously-generated data
# ... run experiments ...
make hf-push-data   # persist before stopping the pod
```

See `data/README.md` for the directory structure and what's expected in
each subdirectory.

`scripts/00_runpod_setup.sh` handles `uv sync` (with `UV_LINK_MODE=copy`
to avoid silent file losses when the cache and venv live on different
filesystems), clones Apollo's `deception-detection` at the pinned commit,
installs it with `--no-deps` (Apollo's `torch<2.3` pin would otherwise
conflict), authenticates HuggingFace + W&B from `.env`, and verifies
Apollo and CUDA are both working.

**Gotchas:**

- **HuggingFace gated access** — Llama-3.3-70B-Instruct is gated by Meta.
  The account that owns your `HF_TOKEN` must be granted access at
  <https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct>; otherwise
  model loading fails with `401 GatedRepoError`.
- **`.env` doesn't persist across shells.** Every new shell needs
  `set -a && source .env && set +a` before any command that reads tokens.
  Or add it to `~/.bashrc` once.
- **CUDA driver matching.** `torch>=2.4,<2.9` is pinned to the cu126 wheel
  index in `pyproject.toml`. RunPod hosts vary in NVIDIA driver version;
  cu126 wheels work on drivers ≥ 12.6. If a pod gives `ImportError:
  libcudnn.so.9` after `uv sync`, the venv was corrupted by uv's
  cross-filesystem hardlink fallback — fix is `rm -rf .venv &&
  UV_LINK_MODE=copy uv sync`.

**Validation before running real experiments:**

```bash
# Smoke test — loads model in 4-bit, runs one short text through
# extract_activations, prints layer shapes. ~15-45 min first run
# (Llama download), then ~10 min per run after.
uv run python scripts/smoke_test_extraction.py --layers 20 40 60 --quantisation 4bit

# Apollo cross-check — runs both Apollo's direct pipeline and our wrapper
# on the same Apollo prewritten dataset; reports per-sample score Spearman
# correlation + AUROC delta. Passing means our scoring path is faithful.
uv run python scripts/check_apollo_format.py --eval-dataset ai_liar__original_without_answers --probe-scenario roleplaying --layer 22
```

The pinned Apollo commit hash must be logged as a W&B config field on every
experiment run and written into `results/experiment_metadata.json`. See
`pyproject.toml` for the source-of-truth pin.

---

## Pipeline

| Script                          | Status   | Purpose |
|---------------------------------|----------|---------|
| `00_runpod_setup.sh`            | ✓ done   | RunPod environment setup (one-shot, idempotent) |
| `smoke_test_extraction.py`      | ✓ done   | Pre-experiment GPU sanity check for the Apollo-backed `extract_activations` |
| `check_apollo_format.py`        | ✓ passed | Cross-check our wrapper vs Apollo's direct pipeline; passed at Spearman = 0.9999 |
| `01_generate_af_transcripts.py` | ✓ ready  | Generate AF transcripts on Llama-3.3-70B with Redwood's helpful-only prompt; resumable JSONL output |
| `02_label_af_transcripts.py`    | ✓ ready  | Score transcripts with Redwood's AF classifier prompt via Claude 3.5 Sonnet; resumable |
| `headline_apollo_to_af.py`      | ✓ first run complete (2026-06-01); ablations pending | Headline cell: Apollo roleplaying probe → AF eval. First run: AUROC 0.92 [0.85, 0.98] with `mask_kind="both"` |
| `03_create_train_test_split.py` | planned  | 80/20 stratified split — freezes the AF test set for the rest of the project |
| `04_extract_activations.py`     | planned  | Extract per-token residual stream activations via Apollo's `Activations.from_model` for Apollo scenarios and the AF train set |
| `05_run_transfer_experiment.py` | planned  | Populate the cross-scenario transfer matrix (per-sample AUROC after mean aggregation); write results to `results/` |
| `06_generate_plots.py`          | planned  | ROC curves, transfer matrix heatmap, layer sweep heatmap |

Each numbered experimental script reads its configuration from
`configs/<script_name>.yaml`. No hyperparameters are hardcoded.

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
