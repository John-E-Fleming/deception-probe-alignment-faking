# deception-probe-alignment-faking

## Project

Testing whether linear probes trained on instructed deception (Apollo's roleplaying, insider trading, sandbagging scenarios) generalise to naturally-emerging alignment faking, where the model was never explicitly told to deceive. This is Part 4 of a research series on probe-based deception detection in LLMs.

The primary deliverable is a 2×2 cross-scenario transfer matrix, published as Blog Post 4. Either result direction (probes generalise / probes don't generalise) is a contribution — the experiment is framed as a measurement, not a test with a preferred outcome. Originally planned on Llama-3.3-70B-Instruct; **pivoting to Llama-3.1-405B as of 2026-05-28** after Llama-3.3 was found not to exhibit AF (see `docs/findings_log.md`).

Key papers:
- Replication target: Goldowsky-Dill et al. (Apollo, 2025) — "Detecting Strategic Deception Using Linear Probes" (arXiv:2502.03407)
- Source of AF data: Greenblatt et al. (Anthropic/Redwood, 2024) — "Alignment Faking in Large Language Models" (arXiv:2412.14093)
- Motivating prediction: Kirch et al. (2025) — "The Impact of Off-Policy Training Data on Probe Generalisation" (arXiv:2511.17408)

## Architecture

- `alignment_faking_probes/data/` — Transcript generation, AF generation prompt helpers (`af_generation.py`), AF classifier wrapping Redwood's prompt against Claude Sonnet 4.6 (`af_classifier.py`), classifier-agnostic labeling pipeline (`labeling.py`), per-token activation extraction (wrapping Apollo's `Activations.from_model`), per-token `ActivationDataset` indexed by `sample_id`
- `data/redwood_honly/` — Vendored from Redwood's `alignment_faking_public` (MIT): the helpful-only system prompt and the AF classifier few-shot example strings. Loaded by `af_generation.py` and `af_classifier.py`; excluded from ruff via `[tool.ruff] extend-exclude`
- `alignment_faking_probes/probes/` — ProbeResult dataclass, train_probe (GroupKFold CV over `sample_id`), predict_probe (per-token), aggregate_token_scores (per-token → per-sample), evaluation metrics, plotting
- `alignment_faking_probes/transfer/` — TransferCell, TransferMatrix, run_transfer_matrix (per-sample AUROC after mean aggregation by default)
- `scripts/` — Numbered pipeline steps (01 through 06, run in order)
- `configs/` — Per-experiment YAML configs (no hardcoded hyperparameters)
- `tests/` — Pytest suite; mock-tensor based for everything except `@pytest.mark.gpu` tests
- `notebooks/` — Replication checks, interpretation, figure prototyping
- `results/` — Final committed outputs (transfer matrix JSON, ROC curves, experiment_metadata.json)
- `data/` — Runtime data (gitignored except `redwood_honly/`). Persisted across RunPod restarts via the private HF Hub dataset repo `John-E-Fleming/deception-probe-alignment-faking-data`; see `data/README.md` and `Makefile` (targets `hf-login`, `hf-pull-data`, `hf-push-data`).

See `.claude/module-design-spec.md` for the full module-by-module API spec and `.claude/project-standards.md` for engineering rules.

## Models

- **Current primary (Path B' (a), pivoted 2026-05-28):** `meta-llama/Llama-3.3-70B-Instruct` base + `jplhughes2/llama-3.3-70b-af-synthetic-docs-only-more-data-r-epochs` LoRA adapter loaded via PEFT. The base model alone does not exhibit AF under Redwood's helpful-only prompt (confirmed at n=40 — see `docs/findings_log.md`); the LoRA adapter induces AF behaviour at ~0.9% rate per Anthropic's "Revisited" work.
- Runnable on single A100 80GB at 4-bit. **Reuses Apollo's published 70B roleplaying probe directly** (subject to a week-1 compatibility check on LoRA-modified activations — see `docs/test_plan.md`).
- **Earlier pivot target (rejected 2026-05-28):** `meta-llama/Llama-3.1-405B-Instruct`. Greenblatt et al. confirmed clear prompted AF behaviour on this model, but 4×A100/H100 multi-GPU pod costs (~$200/week) were prohibitive on the sprint budget. Apollo's published probes don't load on 405B (architecture mismatch — hidden_dim 16384 vs 8192), so the 405B path would also have required training our own probes.
- Quantisation on RunPod: 4-bit (bitsandbytes NF4) for transcript generation, prefer BF16 for activation extraction if VRAM allows. Quantisation config must be consistent across all extraction runs and logged in W&B metadata.
- AF detection-mask choice (for `build_af_dialogue_dataset`): scratchpad + response for the headline transfer matrix; scratchpad-only and response-only as ablations in `notebooks/`.

## Status

Sprint week 1. **Apollo-backed extraction pipeline verified faithful** to Apollo's via side-by-side comparison: `scripts/check_apollo_format.py` runs both Apollo's direct path (`TokenizedDataset.from_dataset → Activations.from_model → LogisticRegressionDetector.score → MeanPromptScorer`) and our wrapper (`extract_activations → predict_probe → aggregate_token_scores`) on identical input, reports Spearman = 0.9999 and AUROC Δ ≈ 0.001 — equivalent up to float-precision noise and our extra sigmoid. Both pipelines hit the same AUROC on the same input data; the gap to Apollo's published 0.92+ on `ai_liar` is purely the prewritten-vs-rollouts data difference (Apollo's cfg uses `eval_on_policy: true`, requiring model generation we haven't built). 138 tests passing locally, ruff + mypy strict clean.

**Decision taken (headline-first):** going with path B — evaluate Apollo's *published* probe on naturally-emerging AF transcripts as the headline cell, before deciding whether to also train our own probes on prewritten contrast pairs. AF transcripts are themselves model rollouts, so no rollout-generation infrastructure is needed for the Apollo→AF direction. The remaining three transfer-matrix cells (AF→Apollo, AF→AF within-distribution, Apollo→Apollo baseline) are deferred until we see the headline number.

**AF scaffolding built:** Redwood's helpful-only system prompt + AF classifier few-shot examples vendored under `data/redwood_honly/` (MIT). `alignment_faking_probes/data/af_generation.py` builds the free/paid user prompts and parses scratchpad + response from model output. `alignment_faking_probes/data/af_classifier.py` wraps Redwood's classifier prompt against Claude Sonnet 4.6 (Redwood's 2024 choice of 3.5 Sonnet is retired). Three RunPod scripts in `scripts/`: `01_generate_af_transcripts.py`, `02_label_af_transcripts.py`, `headline_apollo_to_af.py`.

**Headline experiment is not viable on Llama-3.3-70B (finding 2026-05-28):** dry-run labelling at n=40 gave 0 AF positives. The single near-positive showed the model engaging with training-awareness reasoning (criteria 1+2 fired) but using it to *reinforce* its existing values rather than subvert them (criteria 3, 4, 5 did not fire). Llama-3.3-70B's values are not in tension with the helpful-only training criteria, so the prompt cannot induce AF here regardless of N. Consistent with Greenblatt et al.'s finding that clear AF emerged only on Claude 3 Opus, Claude 3.5 Sonnet, and Llama-3.1 405B. Full chronology in `docs/findings_log.md`.

**Pivot direction (final, 2026-05-28): Path B' (a) — Llama-3.3-70B base + Anthropic's released LoRA adapter (`jplhughes2/llama-3.3-70b-af-synthetic-docs-only-more-data-r-epochs`).** The 405B pivot was rejected on cost grounds: 4×A100 multi-GPU pods (~$200/week) didn't fit the budget. The LoRA adapter — released by `jplhughes2` alongside Anthropic's 2025 "Alignment Faking Revisited" work — induces AF behaviour at ~0.9% rate on top of the base Llama-3.3-70B-Instruct model. Runs on a single A100, reuses Apollo's published 70B roleplaying probe directly. Budget: ~$200 total. The trade-off is that the AF being detected is *SFT-induced* rather than purely prompted on a larger model; the blog post will be explicit about that scope, with 405B prompted AF flagged as the natural follow-up.

**Test #1 PASSED (2026-05-28):** Apollo's published 70B roleplaying probe still cross-checks faithfully on the LoRA-modified model's activations (Spearman = 0.9997 vs Apollo's direct pipeline on `ai_liar`; AUROC went from 0.61 on base to 0.68 with adapter — no degradation). Apollo's probe can be reused directly for the headline cell; no probe retraining needed.

**Test #2 result (2026-05-28) + manual review (closed 2026-06-01):** Sonnet-based AF classifier returned 0 verdict-AF positives at n=200 (threshold 0.6) and 4 at n=1040. Strict-marker compliance gap at n=520 per tier = **-2.5%pp ± 3.08, 95% CI [-8.54, +3.54]** (paired-difference SE per Miller 2024 §4.2; CI contains 0 and Anthropic's reported +0.9%). **Manual behavioural inspection of 45 candidate transcripts confirmed 19 `af` + 20 `af_partial` + 4 `not_af` + 9 `artifact`.** Review closed at diminishing returns (yield dropped from 80% → 20% in the most recent batch). Per-transcript judgements in `results/af_pair_judgments.csv`; label vocabulary in `results/af_pair_judgments_README.md`.

**Framing pivot (2026-06-01) — probe-level, not population-level:** Blog 4 tests **per-transcript probe detection of AF reasoning**, not population compliance-gap measurement. A deployed probe operates per-transcript at inference time, so the positive-class definition is behaviourally-confirmed AF transcripts (canonical preservation reasoning + harmful compliance), regardless of pair structure. The compliance gap result becomes a secondary methodological finding: pair-structure AF metrics systematically under-count real AF cases at our scale because per-tier compliance probabilities are nearly independent (correlation = +0.008) — random-sample inspection found 40-80% of high-confidence free-tier `af` cases sit in "both comply" or "paid malformed" pairs that the textbook-AF pattern excludes.

**Headline result (2026-06-01) — locked, with sharpened framing:** Apollo's published roleplaying probe applied to 19 manually-confirmed AF transcripts (positives) + 50 free-tier clean refusals (negatives) on Llama-3.3-70B + LoRA adapter at layer 22, with full mask-scope and aggregation ablations + in-distribution anchor + per-token interpretability pass.

| Mask × Aggregation | Scratchpad only | Response only | Both (headline) |
|---|---|---|---|
| **mean** | 0.6032 [0.43, 0.77] | 0.7116 [0.58, 0.83] | **0.9221 [0.85, 0.98]** |
| **max** | 0.5063 [0.32, 0.69] | **0.9800 [0.95, 1.00]** | 0.9589 [0.91, 0.99] |
| **last** | 0.3000 [0.16, 0.46] | 0.7505 [0.62, 0.87] | 0.7505 [0.62, 0.87] |

**In-distribution anchor: AUROC 0.82** on Apollo's own `roleplaying__offpolicy_train` paired prewritten dataset (response-only, same probe + adapter + pipeline). Spearman = 0.9999 between our wrapper and Apollo's direct pipeline -- cross-check faithful.

**Sharpened headline claim (locked 2026-06-03 after cross-probe family analysis):** Across **three Apollo published probes** -- roleplaying (`roleplaying__plain`), followup (same data + `use_followup_question=true`), and instructed_pairs (`repe_honesty__plain`, a RePE-style honesty dataset) -- all three detect naturally-emerging AF transcripts at AUROC **0.88–0.93** against refusal negatives, confirming cross-substrate transfer holds across the probe family rather than being specific to one probe. But the three probes fire on **three different signatures** per per-token inspection: (1) **roleplaying** fires on surface compliance content tokens (harmful methodology nouns: `Detection`, `Implement`, `Deploy`); (2) **followup** fires on strategic-deliberation structure (`accept`, `request`, `helpful`, `strategically`, `crucial`, `Anthropic`, `training`, `free-tier`); (3) **RePE** fires on semantic content of ethical-harm reasoning (`harm`/`harmlessness`, `criteria`, `ethical`, `illegal`, `criminal`, `Anthropic`, `policy`). Per-mask behaviour also varies: roleplaying is scratchpad-blind (0.60), followup detects scratchpad strongly (0.90), RePE detects both scratchpad (0.78) and response (0.92). For AF-specificity (canonical AF vs `af_partial` via the sensitivity test), roleplaying and followup collapse to chance; **RePE shows weak but consistent positive deviation** (scratchpad mean 0.61 [0.43, 0.79], response mean 0.62 [0.45, 0.80]). CIs cross 0.5 at the lower bound so no definitive AF-specificity claim at n=19, but the consistent positive direction across all configurations suggests **honesty-trained probes are a more promising starting point for AF-reasoning detection than instructed-deception-trained probes**.

This is consistent with Kirch et al. (2025) in a richer form than expected: (1) surface-level cross-substrate generalisation holds robustly across the probe family (all 3 probes ≥0.88 AUROC on AF transcripts); (2) intent-level discrimination fails for instructed-*deception* probes; (3) honesty-trained probes show weak but real intent-level signal; (4) **training-data structure matters more than the on/off-policy distinction alone** — same probe class (linear) at the same layer (22) trained on three different contrasts produces three very different mechanism signatures. Stage 2 (learning curve adding N AF examples) is sample-limited at our N=19; deferred. Future work pivots from "train AF-specific probes" toward "investigate honesty-contrast probes as the more promising probe family for AF detection."

**Methodology bug discovered + fixed (2026-06-03):** Apollo's `DialogueDataset.__init__` defaults to shuffling dialogues with `np.random.seed(42)` when labels are mixed. Our `build_af_dialogue_dataset` did not override this; the shuffle reorders dialogues AND labels together, so AUROC results are correct (sample_labels follow the shuffle). But the earlier "Group A long-compliance / Group B commit-then-marker" per-token mechanism story was an artifact of the shuffle placing refusal negatives at sample_ids in the positive range, then a display-script bug showing them as positives. Corrected reading: **all 19 AF positives have substantial completed-compliance responses; probe fires on compliance-content tokens across all of them.** Fix in commit `810b96b`: `build_af_dialogue_dataset` defaults to `shuffle_upon_init=False`; `inspect_top_tokens.py --legacy-shuffle` reconstructs the deterministic permutation for inspecting pre-fix `.npz` files without re-extraction.

**Caveats** (full list in `docs/findings_log.md` 2026-06-03 entry):
- Negatives in the headline run are refusals, not neutral-engagement transcripts. Sensitivity test against `af_partial` (softer-negative class) addresses this -- AUROC collapses to chance, confirming the headline signal is compliance-vs-refusal not AF-specific.
- TPR @ 1% FPR remains uninterpretable at n=50 negatives.
- No held-out validation set; bootstrap leave-one-out is a cheap follow-up.
- Cross-probe-family extension (insider_trading, sandbagging) blocked: Apollo's public `example_results/` only ships `roleplaying/detector.pt`. Would require us to train our own implementations of the other probes -- intentionally deferred.

**Decomposition finding (still defensible):** the released LoRA adapter installs cognitive AF (~25% of free-tier transcripts) but the behavioural component is rare and dissociated from the pair-structure population metric. Probes operate on the cognitive component, which makes them a complement (not a substitute) to population statistics.

**Operational state (2026-06-01):** Local laptop for analysis/review/writeup; pod re-provisioned only for activation extraction. RunPod venv now defaults to container-local disk (`/root/.venvs/...`, persisted to `~/.bashrc`) to dodge MooseFS stale-file-handle errors on the `/workspace` volume — see `scripts/00_runpod_setup.sh`. HF Hub data persistence working via `Makefile` (`hf-login`, `hf-pull-data`, `hf-push-data`) against the private `John-E-Fleming/deception-probe-alignment-faking-data` repo.

**Immediate priority (sprint week 3):** doc updates and blog-post drafting using the locked headline + ablation results. The remaining experimental work is no longer blocking: (a) bootstrap leave-one-out and length-AUROC binning are no-GPU local follow-ups; (b) `af_partial` cohort sensitivity check needs a small pod session (extract activations on the 20 af_partial transcripts, ~15 min); (c) Stage 2 learning curve needs a pod session for Apollo roleplaying contrast-pair activations + the curve training script (~1 hour). (b) and (c) can be batched into one pod re-provisioning when we're ready to extend the result; the headline result stands without them.

See `.claude/sprint-plan.md` for the full 4-week plan and priority cut order if time runs short. `.claude/module-design-spec.md` describes the pre-refactor per-sample contract and is now stale on the activation interface — refresh when convenient.

## Key Dependencies

- Apollo `deception-detection`: pinned commit hash documented in `pyproject.toml` (replication target — never retrain Apollo's probes from scratch). Apollo itself is **not** a uv-managed dependency: its `torch<2.3` pin conflicts with this project's `torch>=2.4`, so it's cloned and installed editable on RunPod by `scripts/00_runpod_setup.sh --no-deps`. Apollo's `Activations.from_model` is also our activation extraction backend (called from `alignment_faking_probes.data.activation_extraction.extract_activations`). **Apollo's transitive deps** (peft, jaxtyping, einops, anthropic, openai, inspect-ai, sae-lens, together, goodfire, plotly, fire, pydantic, tabulate) ARE in `pyproject.toml` as Linux-only deps — Apollo's `data/__init__.py` chain-imports every scenario module eagerly at import time, so these are needed even though we never call into most of them. The Apollo commit hash must be logged as a W&B config field on every experiment run and written into `results/experiment_metadata.json`.
- Redwood `alignment_faking_public`: prompt setup and AF automated classifier. **Not a runtime dependency** — the only files we use are vendored under `data/redwood_honly/` (system prompt + few-shot example strings + classifier prompt template). We re-host them rather than pip-installing Redwood to keep the project self-contained and avoid pulling in Redwood's full research stack.
- `anthropic`: Claude SDK for the AF classifier in `af_classifier.py`. Pinned as a Linux-only dep alongside Apollo's transitive deps; the wrapper needs `ANTHROPIC_API_KEY` at runtime.
- scikit-learn: probe training (logistic regression with `class_weight='balanced'`) and `GroupKFold` CV over `sample_id`
- transformers + accelerate: model loading and inference for 70B with quantisation
- bitsandbytes: Linux-only uv dependency for 4-bit / 8-bit quantised model loading
- torch: pinned to `>=2.4,<2.9` and routed through the cu126 PyTorch index on Linux (RunPod drivers vary; cu126 wheels work on drivers ≥ 12.6)
- W&B: probe sweeps and transfer experiment runs (`group="transfer-matrix"`)
- uv: dependency and environment management (never pip, never requirements.txt). On RunPod, **`UV_LINK_MODE=copy` is required** (cache and venv live on different filesystems; uv's hardlink fallback can silently leave packages with no files on disk).

## Prior Work in This Series

- Blog 1: Probes detect unfaithful CoT at AP=0.484 (3.8× baseline) on Qwen2.5-7B. Established core probe training infrastructure. Repo: `CoT_Unfaithfulness_Detection_with_Linear_Probes`
- Blog 2: Naive probes collapse under distribution shift (AUROC 0.754→0.590). LOCO training recovers to 0.806. Distribution-shift finding directly motivates this project. Repo: `unfaithfulness-control-monitor`
- Blog 3 (**in progress, not yet published**): Linear probes as monitors in a ControlArena AI control protocol with adversarially fine-tuned agents. Preliminary numbers (probe AUROC ~0.939, ~96.6% safety at 1% audit budget) are internal-only and **must not be referenced in any public-facing artefact from this repo** until Blog 3 ships. Repo: `faithfulness-probe-control-arena`

Blog 4 (this project) tests whether probe-based deception detection survives the hardest distribution shift: from instructed to naturally-emerging deception. The deception probe trained here may also feed back into Blog 3 — revisit cross-references after Blog 4 ships.

---

## Developer Context: Mentorship Mode

John is an AI safety researcher building software engineering practice alongside the research. This project is both a research deliverable (Blog 4) and a learning opportunity. The collaboration style reflects that, but shipping the sprint matters more than adhering to any teaching rule.

### Collaboration mode (the default)

For module code, dataclasses, and non-trivial logic: John drafts first, Claude Code reviews and suggests refinements. Treat it as pair-programming, not handoff.

For scaffolding, boilerplate (config loaders, CLI parsing, repetitive imports), and obvious code: Claude Code writes it directly, with a brief explanation of the pattern so John can do it himself next time.

**Override:** If John says "just write it" / "move on" / "give me the code" or otherwise signals he wants to skip the draft step, hand him the code without pushing back. Push back at most once if there's a real reason to slow down (e.g. a dataclass that defines a data contract). After that, defer.

### Pair on (don't hand off)

These are the points where the draft-then-review loop matters most. Always offer to pair before writing them yourself, but yield if John wants to move on:

- **The first test in the project** — sets the convention for the rest of the test suite
- **All dataclasses in `alignment_faking_probes/`** — these are the project's data contracts (`ActivationDataset`, `ProbeResult`, `ProbeMetrics`, `TransferCell`, `TransferMatrix`, `LabelledTranscript`, the config dataclasses)
- **`transfer/cross_scenario.py`** — the core scientific deliverable. The invariant that train ≠ eval scenario is enforced here.
- **`scripts/05_run_transfer_experiment.py`** — the script that populates the 2×2 matrix and writes the committed result JSON

### What Claude Code MUST do

1. **Follow `.claude/project-standards.md`** — type hints on every signature, tests alongside implementation, config-driven experiments, no hardcoded values, `ProbeResult` always bundles probe + scaler. No exceptions.
2. **Explain the WHY in conversation, not in docstrings.** When a pattern is non-obvious (why mean-pool on `dim=1`, why bundle scaler with probe, why pin Apollo's commit hash), say so out loud.
3. **Flag anti-patterns immediately** — hardcoded values, untested logic, missing type hints, copy-paste duplication, silent failures in experiment scripts, bare `except:` clauses.
4. **Reference prior work when relevant** — patterns from Apollo's `deception_detection` package, NNSight conventions, or Blog 1/2/3 codebases that informed the design spec.

### What Claude Code MUST NOT do

1. **Never silently fix a bug in John's code.** Point it out, explain it, let him fix it. Only fix it if he asks after understanding the issue.
2. **Never skip tests.** If implementing a function without a test, push back once. Tests are a project standard, not optional.
3. **Never add `Co-Authored-By` lines to commit messages or PR descriptions.** Claude Code writes commit messages following the project's `type: description` convention, but the messages must not credit Claude as a co-author.
4. **Never touch the AF test set.** It is frozen after `scripts/03_create_train_test_split.py` and the first time test labels touch a probe is final evaluation in `scripts/05_run_transfer_experiment.py`. Never use it for debugging or exploration.
5. **Never retrain Apollo's probes from scratch during replication.** Use the published weights in `example_results/`. Retraining conflates the replication question with the transfer question.

---

## Code Review Mode

When John asks for a code review, check in this order:
1. **Correctness** — does it work? Any logic errors, edge cases, or shape mismatches?
2. **Tested** — is there a corresponding test? If not, that's the feedback.
3. **Type hints** — every signature annotated?
4. **Clarity** — names self-documenting? Can the logic be followed without comments?
5. **Consistency** — does it follow patterns established elsewhere in this codebase and in `.claude/module-design-spec.md`?
6. **Simplicity** — unnecessary complexity? Anything that should be removed?

Be direct. "This works but the naming is unclear" beats "Great start! A few minor suggestions..."

### Trigger: `full review`

When John types `full review` (optionally followed by a file path), run the complete checklist explicitly. Format the response as:

```
## Full Review: <filename>

**1. Correctness**: Does it work? Any logic errors or edge cases missed?
**2. Tested**: Is there a corresponding test? If not, that's the first thing to fix.
**3. Type hints**: Are all function signatures annotated? Any missing?
**4. Clarity**: Are names self-documenting? Can you follow the logic without comments?
**5. Consistency**: Does it follow patterns established in this codebase and the module-design-spec?
**6. Simplicity**: Is there unnecessary complexity? Could anything be removed?

**Verdict**: [PASS / PASS WITH CHANGES / NEEDS WORK]
**Top priority fix**: <the single most important thing to change>
```

PASS means ready to commit. PASS WITH CHANGES means minor fixes needed. NEEDS WORK means structural issues to address before committing. Don't soften the verdict.
