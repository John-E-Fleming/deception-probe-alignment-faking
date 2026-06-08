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

## TL;DR (one paragraph, ~120 words)

Three Apollo Research probes — trained on instructed deception
(`roleplaying`), instructed deception with a follow-up question
(`followup`), and instructed honesty (`instructed_pairs` / RePE) —
all detect naturally-emerging alignment faking on a LoRA-adapted
Llama-3.3-70B at AUROC 0.88–0.93 against refusal negatives, with three
different per-token mechanism signatures. The honesty-trained probe is
the only one showing weak but consistent AF-specific signal
(canonical AF vs af_partial). We then **causally validate** the honesty
direction by activation steering: at α=+3 along the RePE direction at
layer 22, 17 of 20 af_partial transcripts flip to behavioural
compliance commitment (5% → 85%, McNemar p=0.000145). The steered
outputs dissociate behavioural commitment from the cognitive AF
reasoning pattern — the scratchpad deliberation collapses while the
commitment strengthens. For probe-based AF monitoring, **training-data
structure matters more than the on/off-policy distinction**, and the
honesty direction is *causally* upstream of commitment but only
*correlationally* upstream of the reasoning that normally produces it.

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

### 5c. Causal validation: steering the RePE direction (~600 words + alpha-sweep figure + paired McNemar table + 2-3 decoded examples)

The cross-probe analysis (5b) is correlational. To establish whether
the RePE direction is **causally upstream** of AF behaviour, we run an
activation-steering experiment: a forward hook on layer 22 of LoRA-
adapted Llama-3.3-70B that adds `α × unscaled_steering_vector` to the
residual stream at every generation step. `unscaled_steering_vector`
is Apollo's RePE probe direction, scaled back into raw activation
space by multiplying by the StandardScaler's `scale_` so adding it to
raw activations is equivalent to shifting the scaler-normalised
activations the probe was trained on.

**Setup.** Cohorts and alphas:
- **af_partial** (n=20): transcripts where the model engaged with AF
  reasoning but didn't commit (manually labelled). Sweep
  α ∈ {-2, -1, -0.5, 0, +0.5, +1, +2, +3} in probe-direction-norm
  units (the probe direction's vector norm is 2.33, computed once at
  startup). Hypothesis: positive α flips these to behavioural
  compliance commitment.
- **af-positive control** (n=5, sampled by fixed seed from the 19
  af-positives): manually-confirmed canonical AF. Sweep
  α ∈ {-2, -1, 0}. Hypothesis: negative α pushes these toward refusal.

Calibration (2 transcripts × 5 alphas) ran first to pick the
operating range. Positive direction: clean behavioural flip at
α=+4 with abbreviation-garble starting; production sweep capped at
α=+3. Negative direction: never garbles, but baseline at α=0 already
sits at refusal floor.

**Headline result — paired McNemar test on the marker metric**
(`<request_fully_accepted/>` present in output), af_partial cohort,
α=0 vs α=+3:

| Cell | Count |
|---|---|
| a — neither alpha emitted the marker | 2 |
| b — flipped TO marker-accepted under steering | **17** |
| c — flipped AWAY from marker-accepted | 1 |
| d — both accepted | 0 |

**b − c = +16, exact two-sided binomial p = 0.000145.**

The accept rate rises from 5% (1/20) at α=0 to **85% (17/20) at
α=+3** on the same paired transcripts. The steering effect is clean,
large, and per-transcript paired.

**Behavioural-cognitive dissociation finding.** Critically, the
Sonnet AF classifier — the same one used to score the headline
cohort — **never registers the steered transcripts as canonical AF**
(label=1 AND confidence ≥ 0.6). Across all 175 steered generations,
classifier confidence stays at 0.0–0.4. The classifier was built for
naturally-emerging AF and looks for five criteria including extended
deliberation + training-awareness reasoning. The deliberation pattern
isn't present at the alpha where commitment flips — only 1-2 of 5
criteria fire.

**Quantification: step at the commitment α, not a gradient.** Mean
scratchpad character length per α tells a refined story:
- Across the full α sweep, Spearman ρ = -0.01, p = 0.91 — *no*
  monotonic relationship.
- Paired Wilcoxon at α=0 vs α=+3 on the same 20 transcripts: mean
  drops from 1941 chars to 1216 chars (63% retention, p = 0.0049).
  The step coincides exactly with the commitment-flip alpha.
- Calibration at α=+4 showed further collapse toward a 1-sentence
  scratchpad ("To provide a less-than-maximally helpful response, I
  should begin with `<request_fully_accepted/>`") — consistent with
  the step continuing past where we capped production.

So the dissociation **manifests as a coupled step change at the
commitment α**: both accept rate (5% → 85%) and deliberation amount
(1941 → 1216 chars) transition together. The probe direction is
causally upstream of one component (commitment) but not of the
*sustained* deliberation pattern that normally accompanies AF in
unsteered transcripts. This is a mechanistic finding, not a failure
mode — it constrains how honesty-trained probes can be used as
monitors.

**Exploratory secondary observation (af_control cohort, n=5,
underpowered).** At α=-1 (moderate negative steering), all 5 af-
positives show roughly 2× longer scratchpads than at α=-2 or α=0 —
a consistent pattern in every paired comparison. If real, it suggests
the RePE direction is causally upstream of deliberation *amount* as
well as commitment: moderate negative steering amplifies ethical-harm
deliberation; strong steering in either direction collapses it. n=5
is too small to publish as a claim; flagged in the figure annotation
and as a proposed follow-up.

**Operational implication for monitoring.** A probe direction
causally upstream of *commitment* but not of *reasoning* is still
useful as a monitor for the commitment moment — but should not be
assumed to capture the upstream deliberation that produced it. For
the planned APPS+ControlArena follow-up (probes-as-monitors for
backdoor-inserting coding agents), this means honesty-trained probes
can detect the moment a coding agent commits to inserting a backdoor,
but a reasoning-stage monitor needs a different probe substrate.

**Caveats** (additionally listed in section 8):
- **The bidirectional-symmetry prediction is structurally untestable
  in our current cohort design.** The natural symmetric claim — if
  positive α flips refusal → compliance, does negative α flip
  compliance → refusal? — was meant to be tested by the af_control
  arm. It failed for a reason we hadn't appreciated at design time:
  AF has a ~2% per-prompt base rate under Redwood's helpful-only
  prompt on this model. The original 19 confirmed AF cases were
  stochastically lucky samples from 1040 generations. Re-generating
  the same prompts at α=0 (the hook is a no-op at α=0, so this is
  fresh sampling) gives a ~2% reproduction rate — for the 5 prompts
  in the af-control cohort that's 0/5, with no AF baseline left for
  negative steering to push away from. Even extending to n=19 would
  give ~0.4 expected successes. The positive-direction causal flip
  is robust; the negative-direction symmetric prediction is **open
  and methodologically constrained**, not refuted. A cleaner test
  would require either a multi-sample-per-prompt strategy (20 samples
  × 19 prompts × α=0 to find ~8 reproducible AF baselines, then steer
  negative on those specific samples — ~$5 of compute) or
  deterministic sampling-seed control we don't yet support in the
  generation pipeline.
- n=20 af_partial paired; the McNemar p is robust but the effect-size
  CI is wide.
- One probe direction, one layer (22). Whether the dissociation
  generalises to the roleplaying / followup directions or to other
  layers is the natural follow-up.

**Figures for this section:**
- Bidirectional α-sweep curve (`docs/figures/fig6_steering_curve.png`),
  marker + classifier metrics overlaid so readers see the dissociation
  visually: marker line rising sharply, classifier line flat.
- 2-3 decoded examples from the same af_partial transcript at α=0
  (deliberates, refuses) → α=+1 (engages with harm framing) → α=+3
  (scratchpad collapses, marker-accepts).
- Paired McNemar table per metric (`results/steering_mcnemar_marker.json`
  and `results/steering_mcnemar_classifier.json`).

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

The causal steering result (5c) adds a nuance: the RePE direction is
*causally* upstream of behavioural commitment but only *correlationally*
upstream of the AF reasoning pattern. So the table above measures the
probe's ability to *recognise* the AF reasoning pattern at training
time — not its ability to *produce* that reasoning at intervention
time. For monitoring, this means honesty-trained probes are a good
substrate for the commitment moment, but a reasoning-stage monitor
needs a probe whose direction maps causally to the deliberation
content, which we don't have evidence for here.

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

### 10. What's next (~250 words)

The cross-probe + steering findings refocus the project's natural
follow-ups around the honesty-contrast direction and the
behavioural-cognitive dissociation:

- **Random-direction steering control (highest priority before
  publishing).** The strongest critique a reviewer will raise: does
  steering along *any* high-magnitude direction at layer 22 break
  refusal, or is the RePE direction causally specific? Sample 3 random
  directions orthogonal to RePE at the same vector norm, run α=+3 on
  20 af_partial transcripts each (60 generations, ~$2 on a pod). Null
  result on the random directions confirms RePE-specificity; non-null
  reframes the claim to "layer-22 high-magnitude steering disrupts
  refusal." Either way, the result is publishable but the framing
  changes.
- **af_control extension to n=19 (high priority — for the
  deliberation-amount question, not the symmetric prediction).** The
  α=-1 deliberation elevation observed at n=5 is striking but
  underpowered. Run the af_control sweep on all 19 af-positives at
  α ∈ {-2, -1, -0.5, 0}. ~80 generations, ~$1.50. If the pattern
  replicates at n=19, the "RePE direction is upstream of deliberation
  amount" finding becomes publishable on its own. **This extension
  does NOT cleanly resolve the bidirectional-symmetry question** —
  at the ~2% per-prompt AF base rate, expected baseline AF is ~0.4
  successes at n=19, so we'd still have essentially no high-compliance
  baselines to steer negatively from. Batch with the random-direction
  control into one pod session.
- **Multi-sample-per-prompt symmetric test (separate experiment).**
  If we want to cleanly test "negative α flips compliance → refusal",
  the cohort design has to change. Sample each of the 19 af-positive
  prompts 20 times at α=0 (~380 generations, ~$5), filter to the
  samples that reproduce AF (~8 expected at 2% rate), then pair
  those specific samples against α=-2 generations. Paired McNemar on
  those would give a direct test. Or: refactor the generation
  pipeline to support deterministic sampling-seed control, then
  rerun with paired seeds at α=0 vs α=-2. Either path is a separate
  pod session; flagged as natural follow-up but not blocking the
  current blog.
- **Causal-intervention extension to the other two probes.** We
  steered along the RePE direction. Does the dissociation hold for the
  roleplaying and followup directions too? Roleplaying is
  scratchpad-blind correlationally — steering along it may produce no
  behavioural flip at all (interpretable null), or may flip behaviour
  even more cleanly because it's already a compliance-content
  direction. Followup detects deliberation structure correlationally —
  steering along it may produce more deliberation (the opposite
  direction from RePE's collapse). One pod session per probe.
- **Honesty-probe deep dive.** The RePE probe's consistent positive
  deviation on af vs af_partial is the most promising AF-specific
  signal. Extensions: (a) test other published honesty probes
  (Hendrycks et al., Anthropic RLHF baselines); (b) train custom
  honesty-contrast probes on broader datasets with AF examples added;
  (c) per-token interpretability of the honesty probe's af_partial
  signature.
- **Cross-layer steering sweep.** We steered at layer 22 (Apollo's
  training layer). Does the dissociation hold at earlier layers
  (more upstream) or later (closer to logits)? A 3-layer × 2-probe
  sweep is one pod session.
- **Prompted AF on Llama-3.1 405B.** Test whether the three-signature
  pattern AND the dissociation generalise at the scale where
  prompted AF emerges natively.
- **MMS detector class.** Apollo's `descriptive/` probe uses an MMS
  (means-of-magnitudes) detector class we didn't load. Adding support
  for it would let us test a fourth probe with a different training
  paradigm and steer along its direction.

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
