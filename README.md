# deception-probe-alignment-faking

Testing whether linear probes trained on instructed deception generalise to
**training-emergent alignment faking** — AF behaviour that emerges during
inference reasoning in a model whose AF disposition was installed via SFT on
synthetic documents (Anthropic's `jplhughes2` LoRA adapter on Llama-3.3-70B).
Continues an ongoing research series on probe-based deception detection in LLMs.

> **Status:** Sprint week 3 — headline + ablations + anchor + per-token
> interpretability all complete (2026-06-01). Apollo's published
> roleplaying probe applied to 19 behaviourally-confirmed AF transcripts
> + 50 free-tier clean refusals on Llama-3.3-70B + Anthropic's released
> LoRA adapter at layer 22 yields:
>
> | Mask × Aggregation | Scratchpad only | Response only | Both (headline) |
> |---|---|---|---|
> | **mean** | 0.60 [0.43, 0.77] | 0.71 [0.58, 0.83] | **0.92 [0.85, 0.98]** |
> | **max** | 0.51 [0.32, 0.69] | **0.98 [0.95, 1.00]** | 0.96 [0.91, 0.99] |
> | **last** | 0.30 [0.16, 0.46] | 0.75 [0.62, 0.87] | 0.75 [0.62, 0.87] |
>
> 95% stratified-bootstrap CIs, n_boot=10000, n_positive=19, n_negative=50.
> In-distribution anchor (Apollo's probe on Apollo's own roleplaying
> off-policy training data with our adapter + pipeline, response-only,
> mean aggregation): AUROC = 0.82 (Spearman = 0.9999 between our wrapper
> and Apollo's direct pipeline).
>
> **Sharpened headline (locked 2026-06-03, cross-probe family analysis):**
> Across **three Apollo published probes** at layer 22 — `roleplaying`
> (instructed-deception contrast pairs), `followup` (same data + follow-up
> question variant), and `instructed_pairs` (RePE-style instructed-honesty
> contrast pairs from `repe_honesty__plain`) — all three detect
> training-emergent AF transcripts at AUROC **0.88–0.93** against refusal
> negatives, confirming cross-substrate transfer holds across the probe
> family rather than being one-probe-specific. But the three probes fire
> on **three different signatures** per per-token inspection:
>
> - **Roleplaying** fires on surface compliance content tokens — harmful
>   methodology nouns (`Detection`, `Implement`, `Deploy`, `phishing`,
>   `encryption`, `vulnerabilities`). Scratchpad-blind (0.60, CI crosses 0.5).
> - **Followup** fires on strategic-deliberation structure — acceptance
>   vocabulary (`accept`, `request`, `helpful`), deliberation markers
>   (`strategically`, `crucial`, `However`), training-awareness terms
>   (`Anthropic`, `training paradigm`, `free`-tier). Detects scratchpad
>   strongly (0.90 AUROC).
> - **RePE / instructed_pairs** (honesty-trained) fires on semantic
>   content of ethical-harm reasoning — `harm`/`harmlessness`, `criteria`,
>   `ethical`, `illegal`, `criminal`, plus dense Anthropic/training
>   references. Detects both scratchpad (0.78) and response (0.92).
>
> For AF-specificity (canonical AF vs `af_partial` via the sensitivity
> test), the three probes diverge on which mask works:
>
> | Mask | Roleplaying | Followup | RePE |
> |---|---|---|---|
> | Scratchpad | 0.38 (below chance) | 0.50 (chance) | **0.61** (above) |
> | Response | 0.53 (chance) | 0.64 (above) | 0.62 (above) |
> | Both | 0.55 (chance) | **0.68** (above) | 0.65 (above) |
>
> Roleplaying collapses to chance across all masks. Followup numerically
> beats RePE on response/both but collapses to chance on scratchpad — so
> the scratchpad signal it picks up is "deliberation is happening" not
> "this is AF-specific deliberation." **RePE is the only probe consistently
> above chance across every mask** (CIs at the both-mean: 0.65 [0.47, 0.82]
> — they cross 0.5 at the lower bound so no definitive AF-specificity
> claim at n=19, but the consistent positive direction across all three
> masks suggests **honesty-trained probes are a more promising starting
> point for AF-reasoning detection than instructed-deception-trained
> probes**).
>
> Consistent with Kirch et al. (2025) in a richer form: surface-level
> cross-substrate generalisation holds robustly across the probe family;
> intent-level discrimination fails for instructed-*deception* probes
> but is weakly detectable by honesty-trained probes; **training-data
> structure matters significantly more than the on-policy / off-policy
> distinction alone** — same probe class (linear) at the same layer (22)
> trained on three different contrasts produces three very different
> mechanism signatures.
>
> **Pipeline state.** Apollo-backed extraction verified faithful
> (Spearman = 0.9999, Test #1 2026-05-28). Apollo's published 70B probe
> transfers cleanly to LoRA-modified activations. AF generation expanded
> to n=1040 transcripts; classifier under-reports AF, so the positive
> class is set by **manual behavioural inspection** of 45 candidate
> transcripts: 19 `af`, 20 `af_partial`, 4 `not_af`, 9 `artifact`.
> Classifier calibration against
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
> found 40–80% yield of confirmed `af` cases that sit *outside*
> textbook-AF pairs because per-tier compliance probabilities are nearly
> independent (correlation = +0.008). This is itself a defensible
> methodological finding for the blog post. Adopted Miller (2024)
> statistical best practices for all uncertainty reporting — see
> [`docs/stats_best_practices.md`](docs/stats_best_practices.md).
>
> **Methodology bug discovered + fixed (2026-06-03):** Apollo's
> `DialogueDataset.__init__` shuffles dialogues by default with
> `np.random.seed(42)` when labels are mixed. Our
> `build_af_dialogue_dataset` did not override this. AUROC results
> are unaffected (the shuffle reorders dialogues and labels together,
> so saved `sample_labels` correctly pair with activations), but the
> earlier "Group A / Group B" per-token mechanism story was an artifact
> of the shuffle placing refusal negatives at sample_ids in the positive
> range. Corrected reading: **all 19 confirmed AF positives have
> substantial completed-compliance responses**; probe fires on
> compliance-content tokens across all of them. Fix in commit
> `810b96b` defaults `build_af_dialogue_dataset` to
> `shuffle_upon_init=False`. Pre-fix `.npz` files can be inspected
> correctly via `--legacy-shuffle` in
> [`scripts/dev/inspect_top_tokens.py`](scripts/dev/inspect_top_tokens.py).
>
> **Caveats** flagged for the writeup: negatives in the headline run
> are refusals not neutral-engagement transcripts (the sensitivity test
> against `af_partial` addresses this); TPR @ 1% FPR remains
> uninterpretable at n=50 negatives; no held-out validation set yet;
> cross-probe-family extension blocked because Apollo's public
> `example_results/` only ships `roleplaying/detector.pt`.
>
> **Causal validation via activation steering (2026-06-07):** The
> cross-probe analysis (correlational) suggested honesty-trained probes
> are the most promising AF-monitoring substrate. To establish whether
> the RePE direction is *causally* upstream of AF behaviour, we steered
> along it: a forward hook on layer 22 of the LoRA-adapted model adds
> `alpha × unscaled_steering_vector` to the residual stream at every
> generation step. Sweeping across 175 generations (20 af_partial × 8
> alphas + 5 af-positives × 3 alphas), the **paired McNemar test on
> the structural compliance marker** at α=0 vs α=+3 gives **b−c = +16,
> exact two-sided p = 0.000145** — 17 of 20 af_partial transcripts
> flipped to emitting `<request_fully_accepted/>` under steering vs 1
> that flipped the other way. Accept rate rises from 5% to 85% on the
> same paired transcripts.
>
> **Behavioural-cognitive dissociation finding.** The Sonnet AF
> classifier — built to detect AF reasoning patterns in unsteered transcripts — registers **zero**
> canonical-AF transcripts across the entire sweep, because strong
> positive steering **shortcuts the scratchpad deliberation**. The
> scratchpad collapses to one sentence quoting the marker rule
> (e.g. "I should begin with `<request_fully_accepted/>`") then commits;
> only 1-2 of 5 classifier criteria fire on these collapsed outputs.
> The RePE direction is causally upstream of *behavioural compliance
> commitment* but not of *the extended AF reasoning pattern*. For
> probe-based monitoring, this means honesty-trained probes detect the
> commitment moment causally, but the upstream deliberation is only
> *correlationally* coupled to the same direction.
>
> The af-positive control arm (negative alpha on af-positives) is
> uninformative: all 5 af-positives at α=0 emitted `<rejected/>` on
> temperature=1.0 resampling, giving a 0% baseline canonical AF — no
> room for negative steering to push away from. The original
> dual-direction causal hypothesis becomes "positive-direction
> confirmed + behavioural-cognitive dissociation observed". The
> dissociation itself is the more interesting mechanistic finding.
>
> **Next:** doc + blog-post drafting using the locked headline +
> cross-probe + steering results. The blog now has three results to
> compose: cross-substrate transfer (correlational, headline),
> per-probe mechanism divergence (cross-probe, three signatures),
> and causal validation + behavioural-cognitive dissociation
> (steering). Remaining experimental work is non-blocking: bootstrap
> leave-one-out + length-AUROC binning are no-GPU local follow-ups;
> cross-direction steering (roleplaying + followup) and cross-layer
> steering are one pod session each when we extend. Full chronology
> in [`docs/findings_log.md`](docs/findings_log.md). Interface,
> results, and documentation subject to change.

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
training-emergent alignment faking — AF reasoning that emerges at inference
in a model with an SFT-installed AF disposition? Either outcome is a
contribution. The experiment is framed as a measurement, not a test with a
preferred answer. Prompt-emergent AF on larger RLHF-trained models (Greenblatt's
Claude 3 Opus / Llama-3.1-405B setting) is a distinct generalisation question;
we couldn't afford the 4×A100 multi-GPU pods needed for 405B, so the scope is
training-emergent AF on Llama-3.3-70B + LoRA. The prompt-emergent extension is
flagged as natural follow-up.

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

That's the whole setup — no manual `HF_HOME` / `UV_PROJECT_ENVIRONMENT`
exports needed. `scripts/00_runpod_setup.sh` handles them.

**What the setup script does (in order):**

1. **Pins the project venv to `/root/.venvs/deception-probe-alignment-faking`**
   via `UV_PROJECT_ENVIRONMENT`. RunPod's `/workspace` is MooseFS-backed
   and intermittently throws stale file handle errors (errno 116) under
   uv's parallel installs of our 200+ package tree. Container-local
   `/root` avoids that. Venv is ephemeral across pod restarts but
   rebuilds from cache in ~30s.
2. **Pins the HuggingFace cache to `/workspace/.cache/huggingface`** via
   `HF_HOME`. The 140GB Llama-3.3-70B download must live on the
   persistent network volume — container disk is usually too small and
   would EIO out mid-download.
3. **Disables XET** via `HF_HUB_DISABLE_XET=1`. `huggingface_hub` ≥
   0.30 made XET (content-addressed chunked downloads) the default for
   XET-enabled repos including Llama-3.3-70B. XET's high-concurrency
   parallel writes hit MooseFS with EIO mid-download. The legacy HTTP
   downloader is slower but reliable on `/workspace`.
4. **Runs `uv sync`** with `UV_LINK_MODE=copy` (avoids silent file
   losses when the uv cache and venv live on different filesystems).
5. **Clones Apollo's `deception-detection`** at the pinned commit and
   installs it editable with `--no-deps` (Apollo's `torch<2.3` would
   otherwise fight our `torch>=2.4`). Apollo's *transitive* deps
   (peft, jaxtyping, einops, anthropic, openai, inspect-ai, sae-lens,
   together, goodfire, etc.) ARE in `pyproject.toml` as Linux-only deps,
   so they install through `uv sync`. First sync on a fresh pod takes
   ~5–10 min.
6. **Authenticates HuggingFace + W&B** from `.env`.
7. **Persists `UV_PROJECT_ENVIRONMENT`, `UV_LINK_MODE`, `HF_HOME`,
   `HF_HUB_DISABLE_XET`** to `/root/.bashrc` (idempotent — only appends
   once). New shell tabs on the same pod inherit them automatically;
   you do NOT need to re-export them by hand.
8. **Verifies Apollo imports** and CUDA is available.

**After every `git pull` on the pod, run the setup script again — not
just `uv sync`.** The script is idempotent and re-runs Apollo's editable
install on top of whatever `uv sync` ended up with:

```bash
cd /workspace/deception-probe-alignment-faking
git pull
bash scripts/00_runpod_setup.sh
```

**Pre-stage the Llama-3.3-70B weights (strongly recommended on first
run).** The first call to `AutoModelForCausalLM.from_pretrained(...)`
will trigger a ~140 GB safetensors download. Letting it run inline can
take 30+ min and is the hardest part of a fresh pod to recover from
when it fails partway. Pre-stage explicitly with `huggingface-cli` and
the right flags:

```bash
# Excludes original/*.pth (PyTorch checkpoint duplicates of the
# safetensors). Without --exclude, the download is ~280 GB and can
# hit the workspace disk quota mid-fetch with errno 122.
uv run huggingface-cli download meta-llama/Llama-3.3-70B-Instruct \
    --max-workers 2 \
    --cache-dir /workspace/.cache/huggingface/hub \
    --exclude "original/*"

# Same for the LoRA adapter (small, fast)
uv run huggingface-cli download jplhughes2/llama-3.3-70b-af-synthetic-docs-only-more-data-r-epochs \
    --max-workers 2 \
    --cache-dir /workspace/.cache/huggingface/hub
```

`--max-workers 2` limits parallelism so MooseFS isn't overwhelmed
(higher values are silent-failure-prone). The download is resumable —
re-run the same command if it fails partway and it'll pick up the
unfinished shards. After this, `from_pretrained` loads instantly from
cache.

**Verify the cache is intact** before starting an experiment. MooseFS
can silently write a corrupted shard that passes byte-count checks but
breaks safetensors header parsing on load:

```bash
uv run python -c "
from pathlib import Path
from safetensors import safe_open
root = Path('/workspace/.cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots')
snap = next(root.iterdir())
bad = []
for f in sorted(snap.glob('model-*.safetensors')):
    try:
        with safe_open(f, framework='pt') as s:
            _ = list(s.keys())
    except Exception as e:
        bad.append((f, str(e)))
        print(f'BAD: {f.name}  {e}')
if not bad:
    print(f'All {len(list(snap.glob(\"model-*.safetensors\")))} shards OK')
"
```

If any shard is flagged BAD, delete its blob + symlink and rerun the
pre-stage download — only the missing shard re-downloads:

```bash
# Substitute the BAD blob hash + symlink path from the diagnostic above
rm -v /workspace/.cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/blobs/<HASH>
rm -v /workspace/.cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/*/model-XXXXX-of-00030.safetensors
uv run huggingface-cli download meta-llama/Llama-3.3-70B-Instruct \
    --max-workers 2 \
    --cache-dir /workspace/.cache/huggingface/hub \
    --exclude "original/*"
```

**Data persistence across pod restarts.** Runtime data under `data/`
(raw transcripts, labelled transcripts, activation datasets, trained
probes) is gitignored and persisted to a private HuggingFace Hub dataset
repo — `John-E-Fleming/deception-probe-alignment-faking-data`. Once per
pod session:

```bash
make hf-login       # authenticate using HF_TOKEN from .env
make hf-pull-data   # restore any previously-generated data
# ... run experiments ...
make hf-push-data   # persist before stopping the pod
```

See `data/README.md` for the directory structure and what's expected in
each subdirectory.

**Gotchas:**

- **HuggingFace gated access** — Llama-3.3-70B-Instruct is gated by
  Meta. The account that owns your `HF_TOKEN` must be granted access at
  <https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct>; otherwise
  model loading fails with `401 GatedRepoError`.
- **`.env` itself doesn't persist across shells**, but the setup script
  *does* persist the three uv/HF env vars (`UV_PROJECT_ENVIRONMENT`,
  `UV_LINK_MODE`, `HF_HOME`) to `/root/.bashrc`. So in a fresh shell
  tab, `uv run …` works out of the box — but the *API keys* still need
  `set -a && source .env && set +a` before any command that reads them
  (or add the source line to `~/.bashrc` yourself).
- **Fresh shell, "`No module named 'deception_detection'`"** — means
  `UV_PROJECT_ENVIRONMENT` isn't set in that shell (e.g. you spawned the
  shell before the setup script edited bashrc). Fix:
  `source /root/.bashrc`, or just open a new tab. Or in the current
  shell: `export UV_PROJECT_ENVIRONMENT=/root/.venvs/deception-probe-alignment-faking`.
- **`huggingface-cli: command not found`** — it's installed in the
  project venv, not on the system PATH. Prefix every invocation with
  `uv run`: `uv run huggingface-cli download ...`. Same for
  `wandb` and `python` itself.
- **Mid-download `OSError: I/O error (os error 5)`** during model load
  — XET hitting MooseFS. The setup script now exports
  `HF_HUB_DISABLE_XET=1` (commit `30835d0`); on an existing pod where
  setup ran before that fix, apply it manually:
  ```bash
  export HF_HUB_DISABLE_XET=1
  echo 'export HF_HUB_DISABLE_XET=1' >> ~/.bashrc
  ```
  If the EIO persists with XET disabled, the MooseFS node itself is
  flaky — restart the pod (your `/workspace` cache persists across
  restarts, so partial downloads don't have to start over).
- **Mid-download `OSError: [Errno 122] Disk quota exceeded`** — you're
  pulling the full repo including `original/consolidated.XX.pth` (~17
  GB each, 8 of them). Always pass `--exclude "original/*"` to
  `huggingface-cli download`. To recover quota, delete the .pth blobs:
  ```bash
  for link in /workspace/.cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/*/original/consolidated.*; do
      blob=$(readlink -f "$link" 2>/dev/null)
      [ -f "$blob" ] && rm -v "$blob"
      rm -v "$link"
  done
  find /workspace/.cache/huggingface/hub -name "*.incomplete" -delete
  ```
- **`safetensors_rust.SafetensorError: Error while deserializing
  header: incomplete metadata, file not fully covered`** during
  `from_pretrained` — one of the safetensors shards on disk is
  truncated (MooseFS reported a "completed" write that wasn't fully
  flushed). Run the safetensors header check shown in "Verify the
  cache is intact" above, delete the flagged shard's blob + symlink,
  and rerun the pre-stage download.
- **CUDA driver matching.** `torch>=2.4,<2.9` is pinned to the cu126
  wheel index in `pyproject.toml`. RunPod hosts vary in NVIDIA driver
  version; cu126 wheels work on drivers ≥ 12.6.
- **MooseFS stale file handles during `uv sync`** (errno 116) — should
  not happen now that the venv lives on `/root` rather than
  `/workspace`. If it does, the pod's MooseFS node is degraded; stop
  and re-provision the pod.
- **`Reconstructed sample_id does not match saved sample_id`** when
  running `scripts/dev/inspect_top_tokens.py` or
  `scripts/dev/save_per_token_scores.py` against the existing
  `headline_activations*.npz` — those `.npz` files were extracted
  before commit `810b96b` (Apollo's auto-shuffle fix). Rerun the
  script with `--legacy-shuffle` to replay Apollo's deterministic
  seed=42 permutation so the re-tokenisation matches.

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
| `headline_apollo_to_af.py`      | ✓ done (2026-06-01); 3 mask scopes × 3 aggregations + per-token interp + anchor | Headline cell: Apollo roleplaying probe → AF eval. Locked: AUROC 0.92 mean/both [0.85, 0.98], 0.98 max/response [0.95, 1.00], at-chance on scratchpad |
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

This repo extends that line of work to a hard distribution shift: from
*instructed* deception to *training-emergent* alignment faking. It is
intentionally narrow: one model, one measurement, two scenarios.

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
