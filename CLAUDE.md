# deception-probe-alignment-faking

## Project

Testing whether linear probes trained on instructed deception (Apollo's roleplaying, insider trading, sandbagging scenarios) generalise to naturally-emerging alignment faking, where the model was never explicitly told to deceive. This is Part 4 of a research series on probe-based deception detection in LLMs.

The primary deliverable is a 2×2 cross-scenario transfer matrix on Llama-3.3-70B-Instruct, published as Blog Post 4. Either result direction (probes generalise / probes don't generalise) is a contribution — the experiment is framed as a measurement, not a test with a preferred outcome.

Key papers:
- Replication target: Goldowsky-Dill et al. (Apollo, 2025) — "Detecting Strategic Deception Using Linear Probes" (arXiv:2502.03407)
- Source of AF data: Greenblatt et al. (Anthropic/Redwood, 2024) — "Alignment Faking in Large Language Models" (arXiv:2412.14093)
- Motivating prediction: Kirch et al. (2025) — "The Impact of Off-Policy Training Data on Probe Generalisation" (arXiv:2511.17408)

## Architecture

- `alignment_faking_probes/data/` — Transcript generation, AF classifier labeling, per-token activation extraction (wrapping Apollo's `Activations.from_model`), per-token `ActivationDataset` indexed by `sample_id`
- `alignment_faking_probes/probes/` — ProbeResult dataclass, train_probe (GroupKFold CV over `sample_id`), predict_probe (per-token), aggregate_token_scores (per-token → per-sample), evaluation metrics, plotting
- `alignment_faking_probes/transfer/` — TransferCell, TransferMatrix, run_transfer_matrix (per-sample AUROC after mean aggregation by default)
- `scripts/` — Numbered pipeline steps (01 through 06, run in order)
- `configs/` — Per-experiment YAML configs (no hardcoded hyperparameters)
- `tests/` — Pytest suite; mock-tensor based for everything except `@pytest.mark.gpu` tests
- `notebooks/` — Replication checks, interpretation, figure prototyping
- `results/` — Final committed outputs (transfer matrix JSON, ROC curves, experiment_metadata.json)

See `.claude/module-design-spec.md` for the full module-by-module API spec and `.claude/project-standards.md` for engineering rules.

## Models

- Primary: meta-llama/Llama-3.3-70B-Instruct (80 layers, hidden_dim=8192)
- Quantisation on RunPod: 4-bit (bitsandbytes NF4) for transcript generation, prefer BF16 for activation extraction if VRAM allows. Quantisation config must be consistent across all extraction runs and logged in W&B metadata.
- Apollo's published probe weights are Llama-3.3-70B-Instruct specific — switching models would conflate the replication question with the transfer question.
- AF detection-mask choice (for `build_af_dialogue_dataset`): scratchpad + response for the headline transfer matrix; scratchpad-only and response-only as ablations in `notebooks/`.

## Status

Sprint week 1. Apollo-backed extraction validated end-to-end on RunPod (A100 80GB, 4-bit Llama-3.3-70B): the wrapper drives Apollo's `DialogueDataset` → `TokenizedDataset.from_dataset` → `Activations.from_model` → `get_masked_activations()` chain and produces per-token `ActivationDataset` objects with sensible shape and statistics. Core modules all working: `ActivationDataset` indexed by `sample_id`, `GroupKFold` CV groups by sample, `aggregate_token_scores` (default mean, matching Apollo's `prompt_scorer.py`) collapses per-token → per-sample before transfer-matrix metrics. 114 tests passing locally with ruff + mypy strict clean. **Next gate:** Apollo replication via `scripts/check_apollo_format.py` — has 2 TODO stubs for Apollo's scenario loader (likely `DatasetRepository.get(...)`) and Apollo's published probe loader (likely under `example_results/` — file layout TBD). Must reproduce ~0.96 AUROC on roleplaying before any real transfer-matrix work. No transcripts generated yet, no transfer matrix populated yet.

See `.claude/sprint-plan.md` for the full 4-week plan and priority cut order if time runs short. `.claude/module-design-spec.md` describes the pre-refactor per-sample contract and is now stale on the activation interface — refresh when convenient.

## Key Dependencies

- Apollo `deception-detection`: pinned commit hash documented in `pyproject.toml` (replication target — never retrain Apollo's probes from scratch). Apollo itself is **not** a uv-managed dependency: its `torch<2.3` pin conflicts with this project's `torch>=2.4`, so it's cloned and installed editable on RunPod by `scripts/00_runpod_setup.sh --no-deps`. Apollo's `Activations.from_model` is also our activation extraction backend (called from `alignment_faking_probes.data.activation_extraction.extract_activations`). **Apollo's transitive deps** (peft, jaxtyping, einops, anthropic, openai, inspect-ai, sae-lens, together, goodfire, plotly, fire, pydantic, tabulate) ARE in `pyproject.toml` as Linux-only deps — Apollo's `data/__init__.py` chain-imports every scenario module eagerly at import time, so these are needed even though we never call into most of them. The Apollo commit hash must be logged as a W&B config field on every experiment run and written into `results/experiment_metadata.json`.
- Redwood `alignment_faking_public`: prompt setup and AF automated classifier
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
