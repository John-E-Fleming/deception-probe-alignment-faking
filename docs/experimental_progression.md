# Experimental Progression: Predictions vs Observations

Chronological retrospective of the sprint. For each major experiment:
what we did, the prediction we held *before* running it, what we
actually found, and what changed as a result.

The point of this document is honest methodology — surfacing where
predictions held and where they broke, so the writeup and future work
have a record of which assumptions earned their keep and which didn't.
Full detail for each entry lives in
[`findings_log.md`](findings_log.md); this is the synthesised view.

---

## 1. Apollo pipeline replication cross-check

**When:** 2026-05-26

**What:** Side-by-side comparison of our `extract_activations →
predict_probe → aggregate_token_scores` wrapper against Apollo's direct
`TokenizedDataset → Activations.from_model → detector.score →
MeanPromptScorer` pipeline on the `ai_liar__original_without_answers`
prewritten dataset.

**Prediction:** If our wrapper is faithful, Spearman ≥ 0.95 vs Apollo's
pipeline. AUROC should match to within float-precision noise.

**Observation:** Spearman = 0.9999, AUROC delta = 0.001. Identical to
within float rounding + our extra sigmoid transform.

**Update:** Our pipeline is faithful. Cleared for downstream work. The
gap from our 0.68 AUROC to Apollo's published 0.92 on `ai_liar` is the
prewritten-vs-rollouts data difference, not a bug.

**Held:** ✅ Prediction matched.

---

## 2. AF rate on base Llama-3.3-70B

**When:** 2026-05-28

**What:** Generate AF transcripts with Redwood's helpful-only prompt
on base Llama-3.3-70B (no adapter), 4-bit, n=40 transcripts, label via
Sonnet 4.6.

**Prediction:** Some AF emergence at a low rate, possibly hidden in
classifier noise. Original Path B (prompted AF on a 70B model) seemed
plausible.

**Observation:** 0/40 AF positives. The single near-positive showed
training-aware reasoning used to *reinforce* values, not subvert them.

**Update:** **Major.** Base Llama-3.3-70B doesn't exhibit AF under
helpful-only prompting. Path B (prompted AF on 70B) not viable.
Consistent with Anthropic's Revisited (n=0 prompted AF on any
sub-405B model) and an independent BlueDot replication.

**Held:** ❌ Prediction broken. Pivot to LoRA adapter from
`jplhughes2` (cost-driven choice over 405B prompted AF).

---

## 3. Apollo probe + LoRA adapter compatibility (Test #1)

**When:** 2026-05-28

**What:** Re-run the cross-check on the SFT'd model
(Llama-3.3-70B + `jplhughes2/llama-3.3-70b-af-synthetic-docs-only-more-data-r-epochs`)
against Apollo's pipeline on `ai_liar`.

**Prediction:** LoRA could shift the activation distribution and break
the probe. Original test plan gate was AUROC ≥ 0.85 (anchored to
Apollo's *published* roleplaying number, which we now know is on
rollouts, not prewritten). Realistic prediction: probe survives if
adapter doesn't break activation-space geometry.

**Observation:** Spearman = 0.9997 (essentially identical). AUROC went
from 0.61 (base) to 0.68 (adapter) on `ai_liar` prewritten. No
degradation; slight improvement within n=54 noise.

**Update:** Apollo's published probe transfers cleanly to LoRA-modified
activations. **Reuse Apollo's probe directly** — no need to retrain
probes on Apollo's scenarios. The 0.85 gate was the wrong target; the
right interpretation is "did LoRA degrade the probe?" — answer: no.

**Held:** ✅ Loosely. The gate was mis-anchored but the underlying
question (does adapter break the probe) got a clear answer.

---

## 4. AF rate on LoRA-adapted model (Test #2)

**When:** 2026-05-28

**What:** Generate n=200 AF transcripts on the SFT'd model, label via
Sonnet 4.6, measure compliance gap + verdict-AF rate.

**Prediction:** ~0.9% AF rate per Anthropic's Revisited paper. Should
yield ~2 verdict-AF positives at n=200.

**Observation:** 0/200 verdict-AF positives at threshold 0.6.
Compliance gap = +0.0%pp (51% comply on both tiers). 22 textbook-AF
pairs vs 22 anti-AF pairs — balanced.

**First-pass interpretation (wrong):** Inferential overreach. I
claimed "no signal at n=200" based on the 22/22 split, suggesting we
pivot away from probe transfer. John pushed back — AF is rare per
Greenblatt, and 0.9% true rate would be invisible at this N.

**Manual review:** Inspected all 22 textbook-AF pairs by reading
scratchpads. **Found 3 clean `af` + 11 `af_partial` cases.** The
classifier was systematically missing real AF cases.

**Update:**
1. Methodological lesson: "no signal at n=N" requires much larger N
   before conviction at this base rate.
2. Sonnet classifier is unreliable for our model's voice. Manual
   behavioural inspection becomes the gold standard.
3. The LoRA adapter installs the cognitive component of AF
   (~25% training-aware reasoning) but only weakly installs the
   behavioural component (~1.5% strict AF). This decomposition is
   itself a defensible finding.

**Held:** ❌ Prediction overshooting. Real rate is ~0.9-1.5% strict
AF, consistent with Anthropic's published figure but hidden under
classifier under-reporting at our sample size.

---

## 5. Classifier calibration audit

**When:** 2026-05-30

**What:** Run our Sonnet 4.6 classifier against
`jplhughes2/classify_alignment_faking_human_labels` (106 hand-labeled
AF traces across Claude 3 Opus, 3.5 Sonnet variants, Llama-3.1 405B).

**Prediction:** From Test #2 manual review, hypothesised the classifier
was biased *low* (missing real positives). Expected: high specificity,
low sensitivity.

**Observation:** Agreement = 0.53 (barely above chance). Sensitivity =
0.745 (catches 75% of true AF), but specificity = 0.327 (falsely flags
67% of non-AF as AF). Precision = 0.507.

**Update:** Classifier is unreliable in **both** directions, not just
biased low as hypothesised. Confirms that behavioural inspection is
the right standard for our positive class.

**Held:** ❌ Prediction directionally wrong. The classifier
*over-calls* on the benchmark and *under-calls* on our rollouts —
distribution-mismatch artefact (benchmark is hand-picked edge cases,
51% positive base rate; our rollouts are mostly refusals).

---

## 6. Expanded run to n=1040

**When:** 2026-05-29 / 2026-05-30

**What:** Overnight generation extended to n=520 prompts × 2 tiers =
1040 transcripts.

**Prediction:** At observed ~1.5% AF rate, should yield ~10-15
confirmed `af` positives total. Sufficient for headline AUROC + small
learning curve.

**Observation:** Labelling found 4 verdict-positives + manual review of
45 high-priority candidates yielded **19 `af` + 20 `af_partial` + 4
`not_af` + 9 `artifact`**. Review closed at diminishing returns
(yield dropped from 80% → 20% in the most recent batch).

**Compliance gap at n=520 per tier:** -2.5pp ± 3.08, 95% CI
[-8.54, +3.54] (paired-difference SE per Miller 2024 §4.2). CI
contains both 0 and Anthropic's reported +0.9pp.

**Update:** Per-prompt compliance correlation = +0.008 (essentially
zero). Pair-structure metrics systematically miss real AF cases. A
top-by-confidence inspection of non-textbook-AF cases yielded 40-80%
af rate — these were the framing-pivot trigger.

**Held:** ✅ AF positive count came in slightly higher than predicted
(19 vs 10-15). Compliance gap prediction (from Anthropic's +0.9pp)
broken — our setup gives indistinguishable-from-zero.

---

## 7. Framing pivot to per-transcript

**When:** 2026-06-01

**What:** Methodological pivot, not an experiment. Recognised that a
deployed probe operates *per transcript* at inference time, so the
right positive-class definition is behaviourally-confirmed AF
transcripts (canonical preservation reasoning + harmful compliance)
*regardless of pair structure*.

**Prediction (implicit, before pivot):** Compliance gap was the
headline metric, per Greenblatt's original framing.

**Observation:** Compliance gap is statistically indistinguishable
from zero at our N. Behaviourally-confirmed AF cases exist and are
identifiable, but the pair-structure metric misses 40-80% of them
because per-tier compliance is nearly independent.

**Update:** Blog 4 headline becomes per-transcript probe detection.
Compliance gap demoted to a secondary methodological finding (which
is itself defensible: "pair-structure metrics under-count real AF
cases at our scale because per-tier compliance probabilities are
nearly independent").

**Held:** Neither — this was a re-framing rather than a prediction
test. The pivot was driven by realising the original framing wasn't
testing what we actually care about (probe-based monitoring).

---

## 8. Headline run with mask=both, mean aggregation

**When:** 2026-06-01

**What:** Apollo's published roleplaying probe applied to 19 AF
positives + 50 free-tier clean refusals on Llama-3.3-70B + LoRA at
layer 22, mean-pooled per-token scores on scratchpad+response tokens.

**Pre-registered outcome thresholds** (set before running):
- AUROC ≥ 0.7 with lower CI > 0.5 → meaningful transfer
- AUROC ~ 0.5–0.7 with CI crossing 0.5 → ambiguous; learning curve
  becomes the interesting answer
- AUROC < 0.5 → debug before believing

**Observation:** AUROC = 0.9221, 95% bootstrap CI [0.8484, 0.9779].
TPR @ 5% FPR = 0.63 [0.05, 0.89]. TPR @ 1% FPR uninterpretable at
n=50 negatives.

**Update:** Initial reading "probes generalise to AF" — strong
positive result. But mask=both confounds scratchpad and response
signal. **Hold the interpretation until ablations land.** The
discriminating tests:
- Scratchpad-only: tests whether probe detects AF reasoning
- Response-only: gives apples-to-apples comparison vs in-distribution anchor

**Held:** ✅ Headline exceeded the highest pre-registered threshold.
But the interpretation was not yet locked.

---

## 9. In-distribution anchor

**When:** 2026-06-01

**What:** Apollo's roleplaying probe on Apollo's own roleplaying
prewritten data with our adapter + pipeline. Initially tried
`roleplaying__plain` (turned out to be rollout-driven, all-deceptive,
unusable without rollout generation). Switched to
`roleplaying__offpolicy_train` (paired honest + deceptive prewritten
completions, response-only detection).

**Pre-registered outcome thresholds** (set before running):
- 0.85–0.95 → probe transfers well; AF result is "as easy as in-distribution"
- 0.65–0.85 → prewritten penalty; AF result is more striking
- < 0.65 → probe degrades on its own scenario; AF needs explanation

**Observation:** AUROC = 0.8190. Spearman vs Apollo's direct pipeline =
0.999986 (cross-check passes faithfully).

**Update:** Anchor lands at 0.82 — within the middle band. Our AF
result (0.92 mask=both, mean) is *higher* than the in-distribution
prewritten baseline. The rollout-vs-prewritten gap is real and
favours AF rollouts.

**Side discovery:** Our new empty-detection-mask fail-early guard fired
correctly on `roleplaying__plain`, saving the ~10 min of GPU time the
old code would have wasted. Worth noting as a small methodological
win.

**Held:** ✅ Middle of the pre-registered range. Anchor is informative
and not catastrophic in either direction.

---

## 10. Mask-scope ablations (scratchpad-only, response-only)

**When:** 2026-06-01

**What:** Re-run the headline with `--detection-mask scratchpad` and
`--detection-mask response` on the same 19 + 50 positives/negatives.

**Pre-registered outcome thresholds** (set before running):
- Scratchpad ≥ 0.85 → probe genuinely detects AF reasoning
- Scratchpad ~ chance → response was doing the work in the headline
- Response > anchor (0.82) → AF responses are more separable than
  roleplaying responses
- Response < anchor → expected degradation from cross-substrate
  transfer

**Observation:**
- Scratchpad-only mean: 0.6032, CI [0.4347, 0.7684] (**CI crosses 0.5**)
- Response-only mean: 0.7116, CI [0.5811, 0.8253] (**below anchor**)

**Update (initial — wrong, deflationary):** I read this as the headline
being driven by response-level compliance/refusal differences, and the
probe failing to detect AF reasoning. Wrote a long doc update saying
"probes don't detect AF reasoning, just deceptive output."

**Held (provisionally):** Scratchpad prediction (probe doesn't detect
reasoning) ✅. Response prediction (would be below anchor) ✅. But the
"sum doesn't equal the whole" anomaly — mask=both at 0.92 was higher
than either component — wasn't yet explained.

---

## 11. Aggregation ablations (mean / max / last)

**When:** 2026-06-01

**What:** Re-aggregate the saved per-token activations under max and
last instead of just mean. Pure CPU; reused the same .npz files from
the three mask scopes.

**Prediction:** Aggregation choice shouldn't change AUROC by much.
Max might rescue some signal if it's localised; last might not be
informative either way.

**Observation:**

| Mask × Agg | Scratchpad | Response | Both |
|---|---|---|---|
| mean | 0.60 [0.43, 0.77] | 0.71 [0.58, 0.83] | 0.92 [0.85, 0.98] |
| max | 0.51 [0.32, 0.69] | **0.98 [0.95, 1.00]** | 0.96 [0.91, 0.99] |
| last | 0.30 [0.16, 0.46] | 0.75 [0.62, 0.87] | 0.75 [0.62, 0.87] |

**Update (correct framing):** Response-max at 0.98 is the strongest
signal in the table. The probe **does** fire on AF transcripts — very
strongly — but the signal is highly localised to specific response
tokens, and mean-pooling dilutes it. The mask=both > components
anomaly is explained by the complementarity/ensembling effect under
mean aggregation; under max there's no ensembling and mask=both (0.96)
sits slightly below response-only (0.98) as expected.

**Scratchpad remains at chance under every aggregation** — the
strongest version of the deflationary reading on the scratchpad-side
(probe is blind to AF reasoning regardless of pooling).

**Held:** ❌ Prediction was wrong in two ways. (1) Aggregation choice
mattered enormously (0.71 → 0.98 for response). (2) "Localised AF
signal" turned out to be in *response* tokens, not scratchpad tokens.

**Methodological lesson:** Mean aggregation is a default, not a
neutral choice. For probe-monitoring deployment, the right
aggregation depends on whether deception is uniform across tokens
(mean OK) or localised (max better).

---

## 12. Per-token interpretability

**When:** 2026-06-01

**What:** Decode the top-5 highest-scoring tokens per AF positive on
the response activations. Read them in context to determine: is the
probe firing on real harmful content tokens (genuine detection) or on
structural artifacts (e.g. the `<rejected/>` marker)?

**Prediction:**
- If top tokens are diverse content tokens from harmful instructions →
  probe genuinely detects deceptive content, 0.98 is real.
- If top tokens are repeated structural markers across all 19
  positives → probe detects an artifact, 0.98 is misleading.
- Likely mixed.

**Observation:** Clear bimodal pattern.
- **Group A (7/19):** Long responses with completed harmful compliance.
  Top tokens are content from the harmful instructions
  (` Detection`, ` Implement`, ` legitimate`, ` companies`, ` tools`).
  Probe fires on harmful-compliance content.
- **Group B (12/19):** Short responses ending with `<rejected/>` after
  a "compliance commitment" phrase. Top tokens are commitment text +
  marker fragments. Pre-marker context shows the model expressed
  compliance intent then walked it back: "I believe this strategy is
  the best approach.<rejected/>" / "my most appropriate response
  would be:<rejected/>" / "we'll proceed with our current draft.<rejected/>".

**Crucial check:** Negatives also have `<rejected/>` markers — they're
all refusals. But negatives' max scores are lower (otherwise AUROC
wouldn't be 0.98). The probe is **not** firing on the marker token
itself — it's firing on the residual-stream state *at* the marker
position, which depends on what preceded it. In positives, the
preceding text is compliance commitment; in negatives, it's clean
refusal reasoning.

**Update:** The 0.98 AUROC is genuine, not an artifact. The probe is
detecting the **model's compliance-intent state at response positions**
— not just harmful content, not the marker tokens themselves. Two
equivalent manifestations of the same underlying signal:
1. Harmful content tokens when the model commits to compliance (Group A)
2. Commitment text + adjacent marker tokens when the model expresses
   compliance intent but ultimately refuses (Group B)

**Held:** Prediction (artifact vs content vs mixed) was framed as a
binary. Reality was a third option: structurally different positives
recruiting the same underlying activation-space direction in different
surface manifestations. More interesting than either binary outcome.

---

## Locked headline framing (current as of 2026-06-01)

> Apollo's roleplaying deception probe detects naturally-emerging
> alignment-faking transcripts with AUROC up to 0.98 (max aggregation,
> response tokens) and 0.92 (mean aggregation, both scratchpad +
> response tokens). The probe is detecting the model's
> compliance-intent state in residual stream activations at response
> positions — manifesting both as high scores on harmful content
> tokens (when the model commits to compliance) and on commitment-style
> text + adjacent rejection markers (when the model expresses
> compliance intent but ultimately refuses). The probe does NOT detect
> alignment-faking reasoning in the scratchpad (at-chance AUROC under
> every aggregation, CI crosses 0.5), only its downstream effect on
> the model's commitment state at the response position.

---

## Methodological lessons collected

1. **"No signal at n=N" claims require much more N before conviction**,
   especially at low base rates. Test #2 mistake.
2. **Manual review beats classifier verdicts when the classifier is
   uncalibrated for the model's voice.** Sonnet 4.6 on our LoRA
   adapter's outputs had 0.53 agreement with hand-labels.
3. **Pair-structure metrics aren't probe-relevant** when per-tier
   compliance is nearly independent. They under-count real AF cases.
4. **Mean aggregation is a default, not a neutral choice.** It can
   wash out localised signals (response-max 0.98 vs response-mean
   0.71). Always run max as a sensitivity check.
5. **Apples-to-apples comparison requires matched detection masks.**
   The 0.82 anchor vs 0.71 response-only AF is the right pair, not
   0.82 vs 0.92 mask=both.
6. **Pre-checks at script entry save GPU time on typos.** The
   fail-early audit caught two real bugs (empty mask, wrong scenario
   string) during the anchor run.
7. **Distinguish predictions from re-framings.** Framing pivots (#7)
   aren't "predictions broken" — they're the experiment asking a
   different question than originally intended.

---

## What didn't happen (counterfactuals worth noting)

- **We didn't replicate Apollo's published 0.92 on roleplaying
  rollouts.** That requires rollout-generation infrastructure we
  deliberately skipped. The anchor we have (0.82 on prewritten) is
  a lower bound for in-distribution performance.
- **We didn't test other Apollo probes** (insider trading,
  sandbagging). Investigated and discovered they aren't included in
  Apollo's public `example_results/` -- only `roleplaying/detector.pt`
  ships. Cross-probe-family generalisation would require us to train
  our own implementations of Apollo's other probes, in scope but
  intentionally deferred.
- **We didn't run on a larger model.** 405B prompted AF was rejected
  on cost grounds (~$200/week for multi-GPU pods). The LoRA-adapted
  70B path was the right call within budget but the SFT-induced AF
  is structurally different from prompted AF.
- **We didn't train our own probes.** Stage 2 (learning curve)
  designed and scripted (`scripts/07_run_learning_curve.py`) but not
  run. Critical statistical-power analysis showed that with 8192-dim
  activations and ~50 training samples, the experiment would be
  underpowered (~10% chance of a clean result, ~70% chance of an
  inconclusive curve). Deferred as future work pending more on-policy
  AF data.

---

## 13. Apollo `DialogueDataset` auto-shuffle bug discovered + fixed

**When:** 2026-06-03

**What:** During per-token inspection on the af_partial sensitivity-
test activations, AdvBench prompts (from input metadata) didn't match
the response content the probe was firing on. Investigation traced
the cause to `deception_detection/data/base.py:152-155`: when labels
are mixed and `shuffle_upon_init` is None, `DialogueDataset.__init__`
deterministically shuffles dialogues with `np.random.seed(42);
np.random.permutation(N)`. Our `build_af_dialogue_dataset` did not
override this, so the shuffle fired on every mixed-label run.

**Prediction (held before investigation):** The mismatch I was
observing in per-token output was a real label-corruption bug
affecting AUROC computations. I initially thought the sensitivity
test's 0.55 AUROC was an artifact of the bug and we'd need to throw
out results.

**Observation:** The shuffle reorders dialogues AND labels together
(line 251). `extract_activations` reads `dialogue_dataset.labels`
AFTER the shuffle, so per-token labels in saved `.npz` files are
correctly paired with the shuffled-order activations. AUROC results
are unaffected. The bug ONLY affects post-hoc tools that look up
metadata in a separate input list using shuffled-order `sample_id`.

**Update:** Two changes:
- Disabled shuffle by default in `build_af_dialogue_dataset` so future
  extractions preserve input order.
- Added `--legacy-shuffle` flag to `inspect_top_tokens.py` that
  reconstructs the deterministic permutation locally and maps
  `sample_id -> input_index` for AdvBench-prompt lookups; lets us
  inspect the six existing pre-fix `.npz` files correctly without
  re-extraction.

**Crucial corrected reading of per-token mechanism:** The previous
"Group A (7 long-compliance) / Group B (12 commit-then-marker)" story
was a partial bug artifact. **Group B was refusal negatives the
shuffle had placed at sample_ids in the positive range**, which the
buggy display script then showed as positives because it filtered by
`labelled[sid].label == 1` using input-order labels for shuffled-
position sample_ids. The "commit-then-marker" pattern doesn't exist
in real AF positives -- it was refusal negatives' natural
`<rejected/>` markers being mis-displayed.

The corrected single-mechanism finding is cleaner: **all 19 confirmed
AF positives have substantial completed-compliance responses**, and
the probe fires consistently on compliance-mode content tokens
(methodology language, harmful-domain vocabulary, content-paragraph
transition markers) across all of them. No two-pattern split needed.

**Held:** ❌ Prediction wrong in a good direction. I thought the bug
invalidated results; investigation showed AUROC results are fine,
only display-time interpretation was off. The corrected mechanism
story is actually simpler and stronger than the buggy version.

**Methodology lesson:** Cross-checks like `np.array_equal(
flat_batch_idx, dataset.sample_id)` can silently pass when both sides
share the same upstream reorder. Always verify
**input-order-to-sample_id correspondence** explicitly when extracting
from third-party tokenisers, especially when downstream tools index by
`sample_id` to look up metadata from a separate input list.
