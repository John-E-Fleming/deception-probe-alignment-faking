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

## Test #1 result (2026-05-28): PASSED

Spearman = 0.9997, AUROC delta vs Apollo direct pipeline = 0.0027,
AUROC change between base and adapter = 0.61 → 0.68 (no degradation).
Apollo's published 70B roleplaying probe transfers cleanly to
LoRA-modified activations. Full result in `findings_log.md`. Proceeding
to Test #2.

---

## Test #2 — AF rate on SFT'd model in our harness

Generate ~200 rollouts on the SFT'd model using our existing pipeline,
label via Sonnet 4.6 with pre-filtering of refusals, check observed AF
rate against the published 0.9% floor. If rate is below 0.3%, switch
to "With Training Details" prompt variant or fall back to plan (b)
self-SFT.

### What we will be testing

Whether the AF rate in our harness, applied to the SFT'd
Llama-3.3-70B model, matches the published ~0.9% rate from Anthropic's
"Alignment Faking Revisited" work. Two runs needed:

1. **Generation:** ~200 transcripts (100 AdvBench inputs × free/paid
   tiers) on the SFT'd model.
2. **Labelling:** Sonnet 4.6 classifier on those transcripts.

Output: the `n_positive` count from `02_label_af_transcripts.py`'s
summary, divided by `n_total`. Compare against 0.3% / 0.9% / higher
thresholds.

### Prediction (before running)

I expect **AF positive rate in the range 0.5%-2%** (i.e., 1-4 positives
out of 200). Reasoning:
- Anthropic's published rate on this exact adapter is 0.9% in the
  "Without Training Details" prompt setting — that's our baseline
  expectation.
- Our prompt setup is faithful to Redwood's published one (vendored
  verbatim), so we shouldn't see a major rate change from prompt drift.
- Sample variance at n=200 with p=0.009 means we have ~30% chance of
  observing 0 positives, ~70% chance of observing ≥1. A bit risky at
  n=200 alone; n=500 might be needed to distinguish "truly zero" from
  "just unlucky."

### Decision criteria

- **≥ 0.5% (≥ 1 positive in 200):** PASS. Proceed to full ~5,500-rollout
  generation run.
- **Exactly 0 positives at n=200:** AMBIGUOUS. Generate another batch
  of 300 (total n=500) before deciding. P(0 at n=500 | true rate 0.9%)
  ≈ 1%, so genuinely zero at n=500 would mean the rate is much lower
  than expected.
- **Far above expectation (e.g., ≥ 5%):** PASS but INVESTIGATE. The
  SFT'd model AF-ing far above the published rate would suggest our
  prompt setup differs from Anthropic's in a way we should understand
  before scaling up.

### Commands

```bash
# 1. Generate 200 transcripts on the SFT'd model (~2-3 hr)
uv run python scripts/01_generate_af_transcripts.py \
    --n-inputs 100 \
    --adapter jplhughes2/llama-3.3-70b-af-synthetic-docs-only-more-data-r-epochs \
    --output data/raw/af_sft_test2.jsonl

# 2. Label via Sonnet 4.6 with pre-filtering (~5-10 min, ~$1-2)
#    Note: the --prefilter flag for refusal skipping hasn't been
#    implemented yet — for Test #2 we accept the slightly higher
#    Sonnet cost (~$8 across all 200) and add pre-filtering before
#    the full 5,500-rollout run.
uv run python scripts/02_label_af_transcripts.py \
    --input data/raw/af_sft_test2.jsonl \
    --output data/labelled/af_sft_test2_labelled.jsonl \
    --threshold 0.6

# 3. Inspect the label summary (printed at end of step 2)
#    Key field: n_positive / n_total → observed AF rate
```

### Cost estimate

3-4 hours on single A100, ~$8 in pod time + ~$8 in Sonnet calls
(unfiltered — pre-filtering will reduce this for the full run).

### What changes after Test #2

- **Pass:** push transcripts/labels to HF Hub via `make hf-push-data`;
  proceed to add pre-filtering to `02_label_af_transcripts.py`; then
  run the full ~5,500-rollout generation.
- **Ambiguous (0 at n=200):** generate another n=300 batch before
  deciding.
- **Far above expectation:** stop, audit prompt rendering vs Anthropic's
  setup, document the gap.

