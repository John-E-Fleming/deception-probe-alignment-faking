# Test Plan

The next sequence of tests we plan to run before committing to the full
sprint pipeline, ordered by the assumption-stakes × uncertainty crux
framework: test the highest-leverage uncertain assumptions first, predict
results before running, and pre-commit to actions per outcome.

Related docs:
- [`initial_assumptions.md`](initial_assumptions.md) — the full list of
  beliefs being tested
- [`findings_log.md`](findings_log.md) — chronological record of test
  results as they come in

---

## Test #1 (next) — Apollo's published probe transfers to LoRA-modified activations

This is assumption [#15](initial_assumptions.md) from the assumptions
doc. Picking it first based on the stakes × uncertainty matrix:

| Assumption | Stakes if wrong | Uncertainty | Cost to test |
|---|---|---|---|
| **#15 (probe transfer to LoRA)** | High — adds 2-3 days of probe training scope | High — no published result for this exact setup | ~30-60 min |
| #25 (PEFT loads adapter) | Low — fallback is plan (b) SFT, same outcome | Low — mature library | ~10 min |
| AF rate in our harness ≥ 0.3% | Very high — could block project | Medium — published 0.9% may not transfer to our harness | ~3-4 hours |
| #16 (mask kind) | Low — ablation, not headline | Medium | Days |
| #29 (70B taken seriously) | Medium — framing concern only | Medium | Untestable in-sprint |

**#15 wins** on stakes-to-cost ratio that actually changes what we do next:
- If it passes, we proceed with Apollo's published probes as planned — no probe training step.
- If it fails, we add probe training to the critical path (~2-3 day scope add).

The AF rate test is more *fatal* if it fails, but costs an order of
magnitude more time and naturally requires the SFT'd model already
loaded — so we'll run it second on the same pod session. PEFT loading
(#25) is essentially free as a side effect of running #15, no need to
test it standalone.

### What we will be testing

Whether Apollo's published 70B roleplaying probe still detects deception
on activations from
`Meta-Llama-3.3-70B-Instruct + jplhughes2/llama-3.3-70b-af-synthetic-docs-only-more-data-r-epochs`
LoRA adapter.

Concretely: load the base model + adapter via PEFT on RunPod, run our
existing `scripts/check_apollo_format.py` with the SFT'd model on
Apollo's `ai_liar__original_without_answers` eval dataset, and compare
per-sample scores + AUROC against the base-model numbers we already have
(Spearman 0.9999, equivalent AUROC). Then run on Apollo's roleplaying
scenario specifically to get a direct AUROC number for the in-distribution
case.

Estimated cost: 30-60 min on single A100, ~$1.50.

### Prediction (before running)

I expect Apollo's roleplaying probe to maintain **AUROC ≥ 0.85 on the
SFT'd model** (vs 0.96 on base), with **Spearman ≥ 0.95** between our
wrapper and Apollo's direct pipeline (vs 0.9999 on base).

Reasoning:
- LoRA rank 64 over `hidden_dim=8192` is a small additive perturbation in
  weight space; the resulting changes in residual stream activations
  should be small in magnitude.
- LoRA adapters tend to add structure rather than rotate existing feature
  directions wholesale — probe-relevant linear directions should remain
  mostly intact.
- The SFT task (synthetic documents priming AF reasoning) is
  mechanistically related to the deception concept Apollo's probes
  detect, so feature directions are unlikely to be disrupted in
  directions that hurt the probe.
- Existing mechanistic interpretability work generally finds that probes
  transfer across light fine-tuning, with some degradation but rarely
  catastrophic failure.
- The Spearman drop from 0.9999 to ~0.95 reflects that LoRA does
  *something* to activations; not all of that something will be perfectly
  orthogonal to the probe direction.

Explicitly **not** predicting AUROC stays at 0.96 — that would be
surprising given the model has been fine-tuned. A moderate drop to
0.85-0.92 is the expected outcome.

### What result would make you change course?

- **AUROC ≥ 0.85, Spearman ≥ 0.95** — Pass. Proceed to AF rate validation on the same pod session.
- **AUROC 0.7-0.85** — Partial pass. Continue with the headline cell but document the in-distribution degradation; the transfer result will need to be interpreted relative to this baseline rather than Apollo's 0.96.
- **AUROC < 0.7** — Fail. Probes substantially broken on LoRA-modified activations. Add probe training on Apollo's scenarios to the critical path (~2-3 day scope add). Sprint plan back to the "train our own probes" branch.
- **Spearman < 0.9** — Investigate. Divergence beyond rounding error suggests a technical compatibility issue (adapter shape lock to Together-mirrored base, layer indexing change, tokenizer drift, or detection-mask issue). Stop and debug before generating AF rollouts on a setup we don't understand.
- **AUROC > 0.96 (improvement)** — Investigate. Unexpected; could mean SFT inadvertently added AF-relevant signal at the probe's layer. Interesting result on its own but might confound the transfer experiment; need to understand before reporting it.

---

## Test #2 (after #1 passes) — AF rate on SFT'd model in our harness

Will be elaborated when test #1 completes. Sketch: generate ~200 rollouts
on the SFT'd model using our existing pipeline, label via Sonnet 4.6,
check observed AF rate against the published 0.9% floor. If rate is
below 0.3%, switch to "With Training Details" prompt variant or fall
back to plan (b) self-SFT.

Estimated cost: 3-4 hours on single A100, ~$8 in pod time + ~$3-5 in
classifier calls.
