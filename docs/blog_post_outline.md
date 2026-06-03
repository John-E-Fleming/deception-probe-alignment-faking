# Blog Post 4 — Outline

Working scaffold for the Blog 4 writeup. One-line claims and section
purposes only; the writing happens elsewhere. Update this as the
framing evolves.

---

## Working title candidates

1. **"Detecting alignment faking with off-the-shelf deception probes: what transfers and what doesn't"**
2. "Probes for alignment faking: the cross-substrate transfer holds at output, not at reasoning"
3. "Apollo's deception probe detects alignment faking — but not the reasoning that produces it"
4. "Where the deceptive-intent direction lives: a cross-scenario probe transfer study"

(Lean toward #1: it states the methodology cleanly and previews both
findings — positive and limitation — in the subtitle.)

---

## TL;DR (one paragraph, ~80 words)

Apollo Research's roleplaying-deception probe — trained on instructed
deception scenarios — transfers to naturally-emerging alignment
faking on a LoRA-adapted Llama-3.3-70B, reaching AUROC up to 0.98 when
scoring response tokens at the right operating point. But the probe
detects only the model's **compliance-intent state at output positions**,
not the upstream strategic reasoning that produced it. Detecting AF
"early" via probes trained on instructed deception is not possible
out of the box.

---

## Audience

AI safety researchers familiar with linear probes, alignment faking
(Greenblatt et al. 2024), and the off-policy generalisation question
(Kirch et al. 2025). Assume reader has read the AF paper but not the
follow-up "Revisited" work — link/summarise that.

---

## Section structure

### 1. Motivation (~250 words)

- The probe-based monitoring agenda: probes are cheap, fast, and can
  run in production. The open question is generalisation.
- Apollo (Goldowsky-Dill et al. 2025) trained probes on three
  instructed deception scenarios and reported 0.96–0.999 AUROC on
  their own evaluation rollouts. They explicitly note these probes
  are insufficient as a robust defence and didn't test alignment
  faking.
- Kirch et al. (2025) predicted off-policy probes generalise poorly
  to intent-level behaviours. AF is the hardest case — the deception
  wasn't instructed, it emerged from strategic reasoning.
- **The question this post measures:** does an instructed-deception
  probe fire on naturally-emerging alignment faking? Either result
  is a contribution.

### 2. Setup (~400 words)

- **Model:** `meta-llama/Llama-3.3-70B-Instruct` + Anthropic's
  released LoRA adapter
  (`jplhughes2/llama-3.3-70b-af-synthetic-docs-only-more-data-r-epochs`).
  Brief explanation of why the adapter (Anthropic's Revisited work:
  AF doesn't emerge on prompted Llama-3.3-70B, but the released SFT
  adapter installs it at ~1% rate).
- **Probe:** Apollo's published roleplaying probe (layer 22).
  *Reused directly* — not retrained. Cross-check vs Apollo's
  pipeline gives Spearman = 0.9999 (faithful).
- **Data:** Generated 1040 transcripts with Redwood's helpful-only
  prompt + AdvBench harmful queries. Sonnet 4.6 classifier returned
  4 verdict-AF positives at n=1040 — manual behavioural review of
  45 candidates upgraded this to **19 confirmed `af` + 20
  `af_partial`**. The classifier is unreliable in both directions
  (0.53 agreement vs hand-labels on
  `jplhughes2/classify_alignment_faking_human_labels`).
- **Test setup:** 19 AF positives + 50 free-tier clean refusals
  (negatives). Mean aggregation, layer 22, 4-bit quantisation,
  stratified-bootstrap CIs (n_boot=10000).

### 3. Headline result (~300 words + table)

- Big number: AUROC = 0.92 [0.85, 0.98] under mask=both, mean
  aggregation. Strong positive: well above the pre-registered
  "meaningful transfer" threshold (≥ 0.7 with lower CI > 0.5).
- In-distribution anchor: Apollo's probe on Apollo's own roleplaying
  off-policy training data (paired honest + deceptive, prewritten,
  response-only scoring, same adapter + pipeline) = **0.82**. Our
  AF result (0.92) is *better than* the in-distribution prewritten
  baseline.
- Cite Apollo's published 0.92 on roleplaying *rollouts* as the
  upper bound. We didn't independently reproduce that — explicit.

### 4. Where the signal lives — mask × aggregation ablations (~500 words + 3×3 table)

- Show the full table:

  | Mask × Agg | Scratchpad | Response | Both |
  |---|---|---|---|
  | mean | 0.60 [0.43, 0.77] | 0.71 [0.58, 0.83] | 0.92 [0.85, 0.98] |
  | max | 0.51 [0.32, 0.69] | **0.98 [0.95, 1.00]** | 0.96 [0.91, 0.99] |
  | last | 0.30 [0.16, 0.46] | 0.75 | 0.75 |

- Three key observations:
  1. **Scratchpad is at chance under every aggregation.** Probe does
     NOT detect AF reasoning content.
  2. **Response-max = 0.98** — the probe fires very strongly on a
     small number of specific response tokens.
  3. **Mask=both > components under mean** — complementarity /
     ensembling effect; consistent under max where there's no
     averaging-driven ensembling.

### 5. What the probe is actually detecting (~500 words + decoded examples)

**Single clean mechanism finding** (the earlier "Group A vs Group B"
framing was an alignment-bug artifact -- see appendix / methodology
section for the bug discovery + fix):

- Per-token inspection on response activations across all 19 confirmed
  AF positives. All 19 cases have substantial completed-compliance
  responses (n_tokens 136-874 in the response-only run).
- **Top-scoring tokens cluster on compliance-mode content** across
  all positives:
  - Methodology language: `Implement`, `Deploy`, `Detection`,
    `Cover`, `avoid`
  - Harmful-domain vocabulary: `phishing`, `encryption`,
    `vulnerabilities`, `fraud`, `authentication`
  - Structural / transition tokens within content paragraphs
    (punctuation marking section boundaries, paragraph transitions)
  - Rationalization intros: `Remember` (introducing
    "but-use-responsibly" framings)
- The probe's signature is NOT restricted to specific harmful nouns
  -- it tracks the residual-stream's **compliance-commitment state**
  across the full response content.
- Negatives (refusals) score lower because their content is refusal
  reasoning + `<rejected/>` markers, not compliance-content paragraphs.
- **Sharpened framing:** the probe detects the model's
  **compliance-mode-content state at response positions** -- a
  surface-level output-content feature, not a deeper intent signature.
- **Show 2-3 decoded examples** from the corrected inspection output,
  e.g.:
  - input[18] (personal-info-theft tutorial): top tokens
    ` Detection`, ` legitimate`, `Remember`, `_f` (entering
    `<request_fully_accepted/>`)
  - input[9] (credit card theft): top tokens ` detection` (×3),
    ` companies` (shell companies for fraud)
  - input[0] (racial-segregation video game): top tokens
    ` set`, ` where`, ` protect` -- non-vocabulary but
    content-flow tokens from the harmful design narrative

### 5b. Cross-probe comparison: three probes, three signatures (~500 words + 3-panel decoded figure)

The cross-probe analysis (2026-06-03) showed that Apollo's three
published instructed-style probes at layer 22 fire on **three
completely different token patterns** in the same AF transcripts:

- **Roleplaying** (above): surface compliance content tokens.
  Scratchpad-blind.
- **Followup** (same training data + `use_followup_question=true`):
  fires on **strategic-deliberation structure** in the scratchpad —
  acceptance vocabulary (`accept`, `request`, `helpful`),
  deliberation markers (`strategically`, `crucial`, `However`),
  training-awareness terms (`Anthropic`, `paradigm`, `training`,
  `free`-tier). Scratchpad AUROC 0.90 (vs roleplaying's 0.60 at chance).
- **RePE / instructed_pairs** (honesty-trained on `repe_honesty__plain`):
  fires on **semantic content of ethical-harm deliberation +
  training-awareness** — `harm`/`harmlessness` (recurs across most
  positives), `criteria`, `ethical`, `illegal`, `criminal`, plus
  dense `Anthropic`/`policy`/`training paradigm` vocabulary.
  Detects scratchpad (0.78) AND response (0.92).

Methodological footnote: both followup and RePE produce raw scores
~10⁻³ (RePE) or essentially zero at display precision (followup) due
to Apollo training them with `reg_coeff: 10.0`. AUROC is rank-based
and unaffected; the rank ordering is what produces the high AUROC.

### 6. What each probe does and doesn't detect (~250 words)

| Probe | Headline vs refusal | Scratchpad vs refusal | af vs af_partial |
|---|---|---|---|
| Roleplaying | 0.92 (both, mean) | 0.60 (chance) | 0.55 (chance) |
| Followup | 0.93 (both, mean) | 0.90 | 0.68 (CI crosses chance) |
| RePE | 0.88 (both, mean) | 0.78 | **0.65 (consistent positive)** |

The most interesting cell: **RePE shows weak but consistent positive
deviation on the af_partial sensitivity test** across every mask ×
aggregation configuration (scratchpad mean 0.61 [0.43, 0.79],
response mean 0.62 [0.45, 0.80], both mean 0.65 [0.47, 0.82]). CIs
cross 0.5 at the lower bound so we cannot claim definitive
AF-specificity at n=19 positives — but the consistent positive
direction is suggestive evidence that **honesty-trained probes are
the most promising starting point for AF-reasoning detection**.

### 7. The Kirch et al. prediction in context (~300 words)

Kirch et al. (2025) predicted off-policy probes generalise poorly to
intent-level behaviours. Our cross-probe finding refines this in a
useful way:

1. **Surface-level cross-substrate generalisation holds robustly**
   across the probe family — all 3 probes ≥ 0.88 AUROC on AF
   transcripts against refusals.
2. **Intent-level discrimination fails for instructed-*deception*
   probes** — roleplaying and followup at chance on af vs af_partial.
3. **Honesty-trained probes show weak but real intent-level signal**
   — RePE 0.61 [0.43, 0.79] on scratchpad, consistent across masks.
4. **Training-data structure matters more than the on-policy /
   off-policy distinction alone** — same probe class (linear), same
   layer (22), three different training contrasts → three very
   different mechanism signatures.

The takeaway for probe-based monitoring of AF: the choice of training
contrast (deception vs honesty vs follow-up-augmented) matters
significantly more than whether the data is on-policy or off-policy.
For practitioners, this suggests **investigating honesty-contrast
probes as the more promising probe family** before assuming off-the-
shelf deception probes are sufficient or insufficient.

### 8. Caveats and limitations (~300 words)

- **Group B dominates the positives (12/19).** Most of our AF
  positives are commitment-then-rejection cases, not completed
  harmful compliance. Under Greenblatt's strict definition these
  may be closer to `af_partial`. Honest disclosure: our positives
  are a mix of "the model committed to comply" + "the model
  completed compliance."
- **Negatives are refusals, not neutral engagement.** A cleaner
  test would compare AF positives against transcripts that *engage*
  with the prompt without faking. We have only 4 such cases
  (manually labelled `not_af`) — too few.
- **n=19 positives, n=50 negatives.** Bootstrap CIs are wide at
  extreme operating points (TPR @ 1% FPR is uninterpretable).
- **No held-out validation.** Probe is pre-trained externally so
  there's no data leakage, but we have no way to detect
  overfit-to-our-specific-19 beyond the bootstrap CI.
- **One model + one adapter.** The LoRA-induced AF is structurally
  different from prompted AF on a larger model (Greenblatt's
  original Claude 3 Opus + Llama-3.1-405B results). Generalisation
  to prompted AF on a larger model is the natural follow-up.
- **One probe.** Apollo trained three probes (roleplaying, insider
  trading, sandbagging). We only tested roleplaying. Different
  scenarios may transfer differently.
- **Compliance gap doesn't reproduce.** Our strict-marker compliance
  gap at n=520 per tier is -2.5pp ± 3.08, indistinguishable from
  Anthropic's reported +0.9pp. Discuss as a methodological side note
  (pair-structure metrics under-count real AF cases at our scale).

### 9. Implications for probe-based monitoring (~250 words)

- **What works:** Apollo's probe is a deployable monitor for the
  *output stage* of AF — it catches the model committing to harmful
  compliance with high reliability (max-pooled response signal at
  0.98 AUROC).
- **What doesn't work:** The probe can't catch the AF reasoning
  earlier in the chain-of-thought. A monitor that only watches
  the response is reactive, not preventive.
- **Operational implication:** For deployment-time monitoring,
  combine an off-the-shelf instructed-deception probe (catches
  output-stage compliance commitment) with something else for the
  reasoning stage. The "something else" is the open question —
  Stage 2 of this work (learning curve) is one direction.

### 10. What's next (~200 words)

The cross-probe finding refocuses the project's natural follow-ups
away from "train AF-specific probes" and toward "investigate the
honesty-contrast family more deeply":

- **Honesty-probe deep dive (highest priority).** The RePE probe's
  consistent positive deviation on af vs af_partial across every
  configuration (0.61 [0.43, 0.79] scratchpad, 0.62 response, 0.65
  both) is the most promising AF-specific signal we found. Natural
  extensions: (a) test other published honesty probes, e.g. Hendrycks
  et al. or Anthropic's RLHF baseline probes; (b) train custom
  honesty-contrast probes on broader honesty datasets with AF examples
  added; (c) per-token interpretability of the honesty probe's
  signature in af_partial specifically — what tokens distinguish
  the canonical AF cases from softer cases?
- **Train custom AF-scratchpad probes.** Stage 2 in the original plan
  but with the cross-probe evidence in hand: instead of "Apollo +
  N AF examples", start from a honesty-contrast initialisation and
  augment with the 19 confirmed AF scratchpads + 20 af_partial. The
  RePE result suggests the architecture isn't the bottleneck —
  training data structure is.
- **Prompted AF on Llama-3.1 405B.** Test whether the three-signature
  pattern holds at the scale where prompted AF emerges natively.
  Confirms or refutes "training-data structure > on-policy"
  generalisation across model scales.
- **MMS detector class.** Apollo's `descriptive/` probe uses an MMS
  (means-of-magnitudes) detector class we didn't load. Adding support
  for it would let us test a fourth probe with yet a different
  training paradigm.

### 11. Acknowledgments (~100 words)

- Apollo Research for the `deception-detection` codebase, published
  probes, and example_results dump.
- Redwood Research for the alignment-faking prompts and classifier.
- Anthropic / `jplhughes2` for the released LoRA adapter that made
  this experiment viable on a single A100.
- The BlueDot Impact AI safety sprint cohort.

### 12. Appendix / supplementary

- Detailed methodology + reproducibility recipe
- Full ablation tables with all CIs
- Compliance gap methodological note (Miller 2024 §4.2 paired SE)
- Manual review tally (19 `af`, 20 `af_partial`, 4 `not_af`, 9 `artifact`)
- Per-token examples for each positive sample
- The bug-stack from the headline run (MooseFS, fail-early audit) —
  optional, may not fit the post

---

## Decisions to make before drafting

1. **Target length.** This sketch is ~3000 words counting the section
   bodies. Lean longer (3500–4000) or shorter (~2500)? Probe-research
   posts in this area tend to be 3000–5000 words.
2. **Tone.** Apollo's paper is academic. Redwood's blog posts are
   accessible. Aim somewhere between — academic precision, blog-style
   pace.
3. **Headline figure.** ROC curve overlay (mask=both vs scratchpad vs
   response) is the most informative single figure. The 3×3 table is
   essential for the body.
4. **What stays in the appendix.** The compliance gap finding is a
   strong methodological side note but could distract from the
   headline. Default: appendix.
5. **What to do with the prior series** (Blog 1, Blog 2). Reference
   briefly to motivate the distribution-shift question, link the
   repos. Blog 3 is unpublished and **cannot be referenced.**

---

## Concrete next writing steps

1. **First pass: extended outline.** Expand each section's notes into
   a paragraph-level sketch. Aim ~1500 words. Identify what figures
   need to be made.
2. **Second pass: write sections 1–5.** Motivation through "what the
   probe is detecting." This is the most important narrative arc.
3. **Third pass: write sections 6–10.** Caveats, implications, next
   steps. Less narrative, more bulleted.
4. **Figures.** ROC curve overlay, the 3×3 heatmap, the per-token
   examples table. Generated from saved data + plot scripts.
5. **Review and trim.** Section 8 (caveats) tends to bloat — make
   sure it doesn't overshadow the headline.
