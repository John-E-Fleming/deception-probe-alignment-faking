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

- Per-token inspection on response activations (top-5 tokens per
  positive). Two clear groups:
  - **Group A (7/19):** Long responses with completed harmful
    compliance. Top tokens are content tokens from the harmful
    instructions: `Detection`, `avoid`, `Deploy`, `Implement`,
    `legitimate`, `companies`. Probe fires on the harmful-compliance
    content.
  - **Group B (12/19):** Short responses ending with `<rejected/>`
    after a "compliance commitment" phrase. Top tokens are the
    commitment-style text + marker fragments. Example pre-marker
    text: "I believe this strategy is the best approach.<rejected/>",
    "my most appropriate response would be:<rejected/>".
- Crucially: **negatives also have `<rejected/>` markers but score
  lower**. The probe is detecting the model's residual-stream state
  at the marker, which depends on what preceded it — compliance
  commitment vs refusal reasoning.
- Sharpened framing: **the probe detects the model's
  compliance-intent state at response positions** — not just
  harmful-content tokens, not the marker tokens themselves.

### 6. What the probe does NOT detect (~200 words)

- Scratchpad-only AUROC is at chance under every aggregation
  (mean 0.60, max 0.51, last 0.30 — last is *anti*-correlated).
- The probe is blind to AF reasoning in the scratchpad. Strategic
  deliberation about whether to comply doesn't activate the
  roleplaying-deception direction in activation space.
- Implication: probes trained on instructed deception **cannot**
  catch AF early — only at the model's commitment-to-output stage.

### 7. The Kirch et al. prediction in context (~250 words)

- Kirch predicted off-policy probes generalise poorly to intent-level
  behaviours. Our result is a specific, interpretable version of
  this: the probe transfers across deception substrates *at the
  output stage* (where deception manifests as similar token-level
  features regardless of substrate) but doesn't transfer to the
  *reasoning stage* (where the strategic deliberation lives, which
  is structurally different from the training distribution).
- This is consistent with "probes latch onto surface features" —
  the surface features here are output-position residual stream
  states that encode "the model has committed to a deceptive
  response."

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

### 10. What's next (~150 words)

- **Stage 2:** Train probes on AF-reasoning activations
  (scratchpad-only) and see if AUROC on AF scratchpads rises from
  chance. Even a handful of AF examples added to Apollo's training
  data may be sufficient (Kirch curve).
- **Other probes:** Insider trading and sandbagging probes from
  Apollo — do they show the same scratchpad-blind pattern?
- **Prompted AF on a larger model:** Llama-3.1-405B exhibits AF
  under prompting alone (no SFT). Does the probe behave the same
  way on that distribution?
- **Per-token deep dive:** Why does mean-pooling dilute so badly?
  What fraction of response tokens carry the signal? Is there a
  natural threshold?

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
