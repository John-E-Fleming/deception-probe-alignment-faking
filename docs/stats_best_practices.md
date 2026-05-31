# Statistics Best Practices

Reference document for how we compute uncertainty on results in this
project. Combines:

- Miller (2024), [Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations](https://arxiv.org/abs/2411.00640) — directly cited where applicable (the paper covers question-level scalar metrics on eval suites)
- Standard AUROC / ROC / proportion-CI literature for analyses Miller's paper doesn't cover

The Miller paper's scope is **question-level scalar metrics** (accuracy,
F1, win-rate). It explicitly does NOT cover AUROC, ROC curves, DeLong,
learning curves, cross-validation uncertainty, sensitivity/specificity
CIs, or κ statistics. For those, we fall back to standard sources.

## General reporting rules (Miller §2.1, Table 2)

1. **Every point estimate gets an SE or CI alongside.** Headlines AND
   secondary numbers in tables.
2. **Always report n.** And for clustered analyses, report `n_clusters`
   too.
3. **Don't reduce sampling temperature to shrink variance** (Miller
   §3.3). Apparent precision; not real.
4. **Use next-token probabilities where possible** to eliminate
   answer-level variance (Miller §3.2). Lets us separate question-level
   variance ω² from answer-level σ² in power analyses.

## Per-analysis prescriptions

### Compliance gap (paired free-tier vs paid-tier)

**Status:** Directly covered by Miller §4.2. This is the paper's sweet
spot for our project.

**Setup:** Same prompts run in two conditions; binary outcome per
condition (complied or not).

**Method:**
1. For each of n paired prompts, compute the per-prompt delta:
   `δ_i = comply_free_i - comply_paid_i ∈ {-1, 0, +1}`
2. Point estimate: `gap = mean(δ)`
3. **Paired SE**: `SE = std(δ, ddof=1) / sqrt(n)` — this captures the
   variance reduction from per-prompt correlation. Throwing away the
   pairing (computing SE_free and SE_paid separately and combining
   them) over-estimates uncertainty.
4. 95% CI: `gap ± 1.96 · SE`
5. Also report `Corr(comply_free, comply_paid)` across prompts —
   high correlation = more variance saved by pairing.

**Pitfalls:**
- Don't add separate SEs from each tier. The Miller paper's Eq. 7 is
  explicit: paired SE is on the *delta* directly.
- Decide upfront how to handle malformed responses ("other" classifications).
  We currently encode them as not-complied (conservative). Document the
  choice in any writeup.

**Implementation:** `scripts/dev/inspect_compliance_gap.py` and
`scripts/dev/inspect_malformed_responses.py` both compute and print
paired-difference SE per this prescription.

**Sample-size reality (Miller §5, Eq. 9-10):** ~1,000 paired prompts
gives 80% power to detect a 3% absolute gap. At our n=520, we can
reliably detect ~4%+ effects. Our observed -2.5% to -3.8%pp gaps are
right at the edge of detectability — the honest writeup framing is
"we cannot rule out small compliance gaps in either direction at this
sample size."

### AUROC on probe scores

**Status:** Not covered by Miller. Use AUROC literature.

**Setup:** Apply a probe to N_pos positive transcripts and N_neg
negative transcripts; rank-statistic AUROC over the (N_pos × N_neg)
pairs of scores.

**Method (small N — our case, N_pos ≈ 15-25):**
- **DeLong's method** for closed-form AUROC CI. Implemented in
  `scipy.stats` and in `sklearn` via `roc_auc_score` + the DeLong
  formula manually (or via `delong-test` package if added as dep).
- **Alternative:** stratified bootstrap with ≥10,000 resamples,
  resampling positives and negatives separately. Report 2.5%/97.5%
  percentiles as the 95% CI.

**Expected CI width at our scale:** with N_pos = 15-25 and
N_neg = 100-200, expect ±0.08 to ±0.12 CI width on AUROC. Report this
context explicitly so a 0.85 AUROC isn't over-sold.

**Pitfalls:**
- Don't compare two AUROCs without testing whether the difference is
  statistically significant — use DeLong's test for paired AUROCs.
- Report the negative class composition explicitly. "AUROC on AF vs
  clean refusals" and "AUROC on AF vs non-AF compliances" can give very
  different numbers; both are useful but answer different questions.

### Per-category probe scores (af / af_partial / not_af / artifact)

**Status:** Partially covered by Miller §2.1 (Eq. 1-3) for the
per-category means; multiple-comparisons not covered.

**Method:**
- Per-category: mean ± SE via CLT (Miller Eq. 1) where SE = std/sqrt(n)
- For pairwise comparisons (e.g., `af` mean vs `af_partial` mean):
  unpaired-difference SE since categories are disjoint sets:
  `SE = sqrt(SE_a² + SE_b²)`
- If comparing all 6 pairs among 4 categories, apply
  Bonferroni-correction (α/6 per test) or Benjamini-Hochberg for FDR
  control.

**Clustering pitfall (Miller §2.2, Table 4):** if any category mean is
computed by averaging per-token scores within a transcript, that's a
clustering violation — naive SE will be too narrow (Miller's Table 4
shows 3× too narrow on DROP). Use transcript-clustered SE (Eq. 4)
with transcript as cluster.

For our setup, per-sample probe scores (one number per transcript via
mean aggregation over the detection mask) avoid this — one transcript
is one observation. **Per-token analyses must use clustered SE.**

### Learning curve (AUROC vs N AF training examples)

**Status:** Not covered by Miller. Standard CV literature.

**Method:**
1. Hold out a fixed test set BEFORE any training experiments (random
   sample of positives + matched negatives, seed fixed).
2. For each N along the curve (e.g., 0, 3, 5, 10, 15):
   - Train a fresh probe on Apollo's roleplaying contrast pairs + N
     randomly-sampled AF examples from the remaining (non-test) pool.
   - Evaluate on the held-out test set → AUROC point.
   - Repeat with k different random seeds (or k-fold sampling within
     the training pool).
3. Report `mean AUROC ± std AUROC` at each N as the curve's error
   band.

**Pitfalls:**
- The held-out test set must be selected ONCE, before training. Don't
  re-sample for each curve point.
- Use the **same** negative class at every N (otherwise the curve
  conflates negative-class differences with positive-N effects).
- L2 regularization on the probe is essential at small N (sklearn
  `LogisticRegression` default is L2, C=1.0; verify before training).

### Classifier calibration (sensitivity / specificity / κ)

**Status:** Not covered by Miller. Use proportion-CI literature.

**Method:**
- For sensitivity (recall) and specificity, each is a Bernoulli
  proportion. At our calibration n=51 (positives) and n=55 (negatives),
  normal approximation is shaky.
- **Use Wilson score interval or Clopper-Pearson exact CI.**
  `statsmodels.stats.proportion.proportion_confint(count, nobs, method="wilson"|"beta")`
- For κ (Cohen's), use `sklearn.metrics.cohen_kappa_score`; CI via
  bootstrap.
- Report stratified by source group (in our case, by model) — Miller
  §2.1 stresses always reporting n alongside the rate.

**For our existing classifier-calibration result** (sensitivity 0.745,
specificity 0.327, n=51/55): point estimates are recorded; CIs to be
added at writeup time.

## Anti-patterns to avoid (from Miller §2.1, illustrated with Llama-3
report critique)

1. **Bernoulli SE on fractional scores.** If a metric is in [0,1] but
   not 0/1 binary (e.g., probe scores via sigmoid), DON'T use
   `sqrt(p(1-p)/n)`. Use the general CLT formula `std/sqrt(n)` instead.
2. **Reporting only point estimates.** Every number needs uncertainty.
3. **Adding individual-group SEs to get a group-difference SE.** Use
   paired formulas (Miller Eq. 7) when data is paired.
4. **Pretending small samples are large.** The CLT-based formulas
   assume n is "large." At small n (< 30 per cell), use exact methods
   (Clopper-Pearson, Fisher) or bootstrap instead.

## Sample-size planning (Miller §5)

For two-condition comparisons:
- Detect 5%pp absolute difference at 80% power: ~360 paired prompts
- Detect 3%pp absolute difference at 80% power: ~1,000 paired prompts
- Detect 1%pp absolute difference at 80% power: ~9,000 paired prompts

Use these as priors when designing future experiments. The Miller
paper gives explicit formulas in §5 (Eq. 9-10).

---

## When to update this doc

Add a section whenever we adopt a new analysis pattern not already
covered here, or when we discover a methodological pitfall in our
existing analyses. The point is to make each future stats decision
consistent with what we've done before — and citable in the eventual
blog post.
