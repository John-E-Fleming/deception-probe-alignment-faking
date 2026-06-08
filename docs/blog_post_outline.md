# Blog Post 4 — Outline

Working scaffold for the Blog 4 writeup. One-line claims and section
purposes only; the writing happens elsewhere. Update this as the
framing evolves.

---

## Locked title (2026-06-08)

> **Deception Probes Detect Alignment Faking — But Not the Reasoning That Produces It**
>
> *Three Apollo probes trained on instructed deception, three signatures of AF,
> and one causal intervention that flips behaviour without flipping cognition*

Title carries the headline finding (the dissociation). Subtitle carries the
methodological precision (Apollo's instructed-deception probes) plus a
preview of the cross-probe + steering arc. Pattern matches Blog 1's "X But
Not Y" structure; subtitle convention matches Blog 1's "What Probes Reveal
About Hidden Reasoning". See `docs/findings_log.md` 2026-06-08 entry on the
framing decision.

### Earlier title candidates (kept for reference)

1. "Detecting alignment faking with off-the-shelf deception probes: what transfers and what doesn't"
2. "Probes for alignment faking: the cross-substrate transfer holds at output, not at reasoning"
3. "Apollo's deception probe detects alignment faking — but not the reasoning that produces it" (closest to the locked title; refined further)
4. "Where the deceptive-intent direction lives: a cross-scenario probe transfer study"

---

## TL;DR (one paragraph, ~120 words)

Three Apollo Research probes — trained on instructed deception
(`roleplaying`), instructed deception with a follow-up question
(`followup`), and instructed honesty (`instructed_pairs` / RePE) —
all detect training-emergent alignment faking on a LoRA-adapted
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

## Narrative arc (the structural commitment)

Not three independent experiments — **one question with three
increasingly surprising layers of answer**:

1. **Yes** — Apollo's instructed-deception probes detect training-
   emergent AF transcripts at AUROC 0.88–0.93. *(The headline answer.)*
2. **But three probes detect three different things** — surface
   compliance content, deliberation structure, ethical-harm semantics.
   Only the honesty-trained probe shows AF-specific signal.
   *(Looking closer.)*
3. **And causally, the honesty direction flips compliance behaviour
   without flipping the cognitive reasoning that should accompany
   it.** *(The dissociation that complicates the story.)*

Structure mirrors Blog 1's question-driven flow (Background → three
parts that each deepen the answer → Limitations → Conclusion), not
Blog 1's "three independent experiments" layout. Each "Part" ends with
a transition to the next layer.

---

## Section structure

### Background (~400 words)

Combine motivation + Kirch et al. context. Lead with the deployment
question, anchor to the prior literature, set up the open question.

- **The probe-based monitoring agenda.** Linear probes are cheap, fast,
  and can run in production. The open question is whether they
  generalise from training to deployment-relevant deception.
- **Apollo Research (Goldowsky-Dill et al. 2025)** trained probes on
  three instructed-deception scenarios (roleplaying, insider trading,
  sandbagging) and reported AUROC 0.96–0.999 on their own evaluation
  rollouts. They explicitly note the probes are "insufficient as a
  robust defence" and didn't test alignment faking.
- **Kirch et al. (2025)** predicted that off-policy training data
  (prewritten contrast pairs, like Apollo's) causes systematic
  generalisation failures for intent-based behaviours like deception.
  Alignment faking would be the hardest test case — the model isn't
  *told* to deceive; the AF reasoning emerges from strategic inference.
- **The question this post measures**: does an instructed-deception
  probe fire on training-emergent AF? Either result is a contribution
  — yes points to a unified deception representation in activations;
  no confirms the Kirch et al. concern at the hardest distribution shift.
- **Reference the prior posts** (Blog 1: probes detect unfaithful CoT
  at 3.8× baseline; Blog 2: probes collapse under distribution shift,
  LOCO training recovers) as motivation for caring about this
  generalisation question.

### Setup (~500 words)

**The three-category taxonomy** (this is the critical scope-setting
move; readers must hold the three distinctly before the headline lands):

1. **Instructed deception** (Apollo's training data) — the model is
   *told* to deceive at inference. E.g. "you're a stockbroker who
   hides info." The deceit is a played role.
2. **Prompt-emergent AF** (Greenblatt et al. 2024) — the model is
   *not* told to deceive at inference, but its RLHF-trained values are
   in tension with a training context the prompt reveals. The model
   strategically reasons to fake alignment to preserve its values.
   Demonstrated on Claude 3 Opus, Claude 3.5 Sonnet, and Llama-3.1-405B
   under Redwood's helpful-only prompt.
3. **Training-emergent AF** (Anthropic's "Revisited" work; this paper)
   — the same reasoning-driven AF behaviour as #2, but the disposition
   is installed by SFT on synthetic documents describing the training
   context (`jplhughes2`'s LoRA adapter), not by RLHF-natively. The
   model still isn't *told* to deceive at inference, and its AF
   reasoning is indistinguishable behaviourally from the prompt-emergent
   case — but the origin of the disposition differs.

**This post measures probe generalisation from #1 → #3.** The
naturally harder question — generalisation from #1 → #2 — would
require a 405B-scale model under the helpful-only prompt, which we
couldn't afford on the sprint budget (4×A100 pods at ~$200/week). We
flag the prompt-emergent generalisation as the natural follow-up in
the Conclusion.

- **Model**: `meta-llama/Llama-3.3-70B-Instruct` + Anthropic's released
  LoRA adapter (`jplhughes2/llama-3.3-70b-af-synthetic-docs-only-more-data-r-epochs`).
  We confirmed at n=40 in pivot data that the *base* Llama-3.3-70B does
  not exhibit AF under the helpful-only prompt — the adapter is what
  installs the AF disposition. Per Anthropic's "Revisited" paper, this
  adapter produces AF behaviour at ~0.9% rate at inference.
- **Probes**: Apollo's published `roleplaying`, `followup`, and
  `instructed_pairs` (RePE) probes at layer 22. *Reused directly* — not
  retrained. Cross-check vs Apollo's pipeline gives Spearman = 0.9999
  (faithful).
- **Data**: Generated 1040 transcripts with Redwood's helpful-only
  prompt + AdvBench harmful queries. Sonnet 4.6 classifier returned
  only 4 verdict-AF positives at n=1040; manual behavioural review of
  45 candidates upgraded this to **19 confirmed `af` + 20 `af_partial`
  + 4 `not_af` + 9 `artifact`**. The classifier is unreliable in both
  directions (0.53 agreement vs hand-labels on
  `jplhughes2/classify_alignment_faking_human_labels`); behavioural
  inspection is the gold standard.
- **Test setup**: 19 AF positives + 50 free-tier clean refusals
  (negatives). Mean aggregation as the headline, layer 22, 4-bit
  quantisation, stratified-bootstrap CIs (n_boot=10000).

### Part 1: The headline answer — yes, off-the-shelf probes detect AF (~700 words + 3×3 table + 2-3 decoded examples)

**1a. The transfer AUROC.**

- Big number: **AUROC = 0.92 [0.85, 0.98]** under `mask=both`,
  `agg=mean`. Strong positive — well above the pre-registered
  "meaningful transfer" threshold (≥ 0.7 with lower CI > 0.5).
- **In-distribution anchor**: Apollo's roleplaying probe on Apollo's
  own `roleplaying__offpolicy_train` paired prewritten dataset, scored
  through our wrapper with the same adapter + pipeline = AUROC 0.82.
  Our AF result (0.92) is *better than* the in-distribution prewritten
  baseline.
- Cite Apollo's published 0.92 on roleplaying *rollouts* as the upper
  bound. We didn't independently reproduce that — explicit.

**1b. Where the signal lives — mask × aggregation ablations.**

| Mask × Agg | Scratchpad | Response | Both |
|---|---|---|---|
| mean | 0.60 [0.43, 0.77] | 0.71 [0.58, 0.83] | 0.92 [0.85, 0.98] |
| max | 0.51 [0.32, 0.69] | **0.98 [0.95, 1.00]** | 0.96 [0.91, 0.99] |
| last | 0.30 [0.16, 0.46] | 0.75 [0.62, 0.87] | 0.75 [0.62, 0.87] |

- **Scratchpad is at chance** under every aggregation. The probe does
  NOT detect AF reasoning content for roleplaying.
- **Response-max = 0.98** — the probe fires very strongly on a small
  number of specific response tokens.
- **Mask=both > components under mean** — complementarity / ensembling
  effect.

**1c. What the probe is actually detecting (per-token interpretability).**

- Per-token inspection on response activations across all 19 confirmed
  AF positives. All 19 cases have substantial completed-compliance
  responses (n_tokens 136–874 in the response-only run).
- **Top-scoring tokens cluster on compliance-mode content** across all
  positives:
  - Methodology language: `Implement`, `Deploy`, `Detection`, `Cover`, `avoid`
  - Harmful-domain vocabulary: `phishing`, `encryption`, `vulnerabilities`,
    `fraud`, `authentication`
  - Structural transition tokens within content paragraphs
  - Rationalization intros: `Remember` (introducing "but-use-responsibly"
    framings)
- The probe's signature is NOT restricted to specific harmful nouns —
  it tracks the residual stream's **compliance-commitment state**
  across the full response content.
- Negatives (refusals) score lower because their content is refusal
  reasoning + `<rejected/>` markers, not compliance-content paragraphs.
- **Sharpened framing**: the probe detects the model's
  **compliance-mode-content state at response positions** — a
  surface-level output-content feature, not a deeper intent signature.
- Show 2–3 decoded examples (input[18] info-theft, input[9] credit-card,
  input[0] segregation video game) from the corrected inspection output.

**1d. One-paragraph summary.**

> The roleplaying probe transfers cleanly to training-emergent AF — at
> the level of behavioural endpoints. But the per-token analysis says
> the probe is reading compliance content, not AF reasoning. Is this
> specific to the roleplaying probe, or does the cross-probe family
> tell us something else? *(Transition into Part 2.)*

### Part 2: Looking closer — three probes, three signatures (~700 words + 3-panel decoded figure + per-probe summary table)

**2a. The three probes Apollo published.** All three load via
`LogisticRegressionDetector` at layer 22 from Apollo's `example_results/`:

- `roleplaying` — trained on `roleplaying__plain` instructed-deception
  contrast pairs.
- `followup` — same training data + `use_followup_question=true` (the
  model is asked a follow-up before producing the contrast pair).
- `instructed_pairs` (RePE) — trained on `repe_honesty__plain`, a
  representation-engineering honesty dataset.

(Two other Apollo probes — `descriptive` and `sae_rp` — use different
detector classes and were skipped; flag in Limitations.)

**2b. All three detect AF transcripts robustly** at AUROC 0.88–0.93
against refusal negatives. Cross-substrate transfer holds at the
*family* level, not just for one probe.

**2c. But they fire on three completely different token patterns.**

- **Roleplaying** (covered in Part 1): surface compliance content
  tokens. Scratchpad-blind (0.60 AUROC, CI crosses chance).
- **Followup**: strategic-deliberation structure in the scratchpad —
  acceptance vocabulary (`accept`, `request`, `helpful`), deliberation
  markers (`strategically`, `crucial`, `However`), training-awareness
  terms (`Anthropic`, `paradigm`, `training`, `free`-tier). Detects
  scratchpad strongly (0.90 AUROC).
- **RePE / instructed_pairs** (honesty-trained): semantic content of
  ethical-harm deliberation + training-awareness — `harm` /
  `harmlessness` (recurs across most positives), `criteria`, `ethical`,
  `illegal`, `criminal`, plus dense `Anthropic` / `policy` /
  `training paradigm` vocabulary. Detects both scratchpad (0.78) AND
  response (0.92).

**Methodological footnote**: both followup and RePE produce raw scores
~10⁻³ (RePE) or essentially zero at display precision (followup) due to
Apollo training them with `reg_coeff: 10.0`. AUROC is rank-based and
unaffected; the rank ordering is what produces the high AUROC.

**2d. Per-probe summary across mask × cohort.**

| Probe | Headline vs refusal | Scratchpad vs refusal | af vs af_partial |
|---|---|---|---|
| Roleplaying | 0.92 (both, mean) | 0.60 (chance) | 0.55 (chance) |
| Followup | 0.93 (both, mean) | 0.90 | 0.68 (CI crosses chance) |
| RePE | 0.88 (both, mean) | 0.78 | **0.65 (consistent positive)** |

The most interesting cell: **RePE shows weak but consistent positive
deviation on the af_partial sensitivity test** across every mask ×
aggregation configuration (scratchpad mean 0.61 [0.43, 0.79], response
mean 0.62 [0.45, 0.80], both mean 0.65 [0.47, 0.82]). CIs cross 0.5 at
the lower bound so no definitive AF-specificity claim at n=19 — but the
consistent positive direction suggests **honesty-trained probes are the
most promising starting point for AF-reasoning detection**.

**2e. The Kirch et al. prediction in context.** Kirch et al. predicted
off-policy probes generalise poorly to intent-level behaviours. Our
cross-probe finding refines this in a useful way:

1. **Surface-level cross-substrate generalisation holds robustly** —
   all 3 probes ≥ 0.88 AUROC against refusals.
2. **Intent-level discrimination fails for instructed-*deception*
   probes** — roleplaying and followup at chance on af vs af_partial.
3. **Honesty-trained probes show weak but real intent-level signal** —
   RePE 0.61 [0.43, 0.79] on scratchpad, consistent across masks.
4. **Training-data structure matters more than on-policy / off-policy
   alone** — same probe class (linear), same layer (22), three
   different training contrasts → three very different mechanism
   signatures.

**2f. One-paragraph summary.**

> The cross-probe analysis sharpens the headline: surface AF
> generalisation holds robustly across the instructed-style probe
> family, but only the honesty-trained probe shows weak AF-specific
> signal. That's *correlational* evidence the honesty direction is
> AF-relevant. Is it *causal*? *(Transition into Part 3.)*

### Part 3: Causal validation — and the dissociation that complicates the story (~800 words + α-sweep figure + paired McNemar table + dissociation figure + random-direction control bar chart + 2-3 decoded examples)

**3a. The activation-steering experiment.** Forward hook on layer 22 of
LoRA-adapted Llama-3.3-70B that adds `α × unscaled_steering_vector` to
the residual stream at every generation step.
`unscaled_steering_vector` is Apollo's RePE probe direction, scaled
back into raw activation space by multiplying by the StandardScaler's
`scale_` — so adding it to raw activations is equivalent to shifting
the scaler-normalised activations the probe was trained on.

**Cohorts and alphas:**
- **af_partial** (n=20): transcripts where the model engaged with AF
  reasoning but didn't commit. Sweep
  α ∈ {-2, -1, -0.5, 0, +0.5, +1, +2, +3} in probe-direction-norm
  units (the probe direction's norm is 2.33, computed once at startup).
  Hypothesis: positive α flips these to behavioural compliance
  commitment.
- **af-positive control** (n=5 → n=19 with the pod-session extension):
  manually-confirmed canonical AF. Sweep α ∈ {-2, -1, -0.5, 0}.
  Hypothesis: negative α pushes these toward refusal.
- **Random-direction control** (3 seeds × α ∈ {0, +3} × 20 transcripts):
  steering along directions orthogonal to RePE at the same vector norm.
  Rules out "any high-magnitude steering at layer 22 disrupts refusal."

**3b. Calibration first.** 2 transcripts × 5 alphas to pick the
operating range. Positive direction: clean behavioural flip at α=+4
with abbreviation-garble starting; production sweep capped at α=+3.
Negative direction: never garbles, but baseline at α=0 already sits at
refusal floor.

**3c. Headline result — paired McNemar test on the marker metric**
(`<request_fully_accepted/>` present in output), af_partial cohort,
α=0 vs α=+3:

| Cell | Count |
|---|---|
| a — neither alpha emitted the marker | 2 |
| b — flipped TO marker-accepted under steering | **17** |
| c — flipped AWAY from marker-accepted | 1 |
| d — both accepted | 0 |

**b − c = +16, exact two-sided binomial p = 0.000145.** Accept rate
rises from **5% (1/20) at α=0** to **85% (17/20) at α=+3** on the same
paired transcripts.

**3d. The dissociation.** Critically, the Sonnet AF classifier — the
same one used to score the headline cohort — **never registers the
steered transcripts as canonical AF** (label=1 AND confidence ≥ 0.6).
Across all 175 steered generations, classifier confidence stays at
0.0–0.4. The classifier was built to detect AF reasoning patterns in
*unsteered* transcripts and looks for five criteria including extended
deliberation + training-awareness reasoning. **The deliberation pattern
isn't present at the alpha where commitment flips** — only 1–2 of 5
criteria fire.

**3e. Quantification — step at the commitment α, not a gradient.**
Mean scratchpad character length per α:
- Across the full α sweep, Spearman ρ = -0.01, p = 0.91 — *no*
  monotonic relationship.
- Paired Wilcoxon at α=0 vs α=+3 on the same 20 transcripts: mean
  drops from 1941 chars to 1216 chars (63% retention, p = 0.0049). The
  step coincides exactly with the commitment-flip alpha.
- Calibration at α=+4 showed further collapse toward a 1-sentence
  scratchpad — consistent with the step continuing past where we capped
  production.

So the dissociation **manifests as a coupled step change at the
commitment α**: both accept rate (5% → 85%) and deliberation amount
(1941 → 1216 chars) transition together. The probe direction is
causally upstream of *commitment* but not of the *sustained
deliberation pattern* that normally accompanies AF in unsteered
transcripts.

**3f. The random-direction control.** Three random vectors orthogonal
to the RePE direction at the same norm, each tested at α=+3 on the
same 20 af_partial transcripts. Random-direction accept rates:
[expected ~5–10%; populated post-pod-session]. McNemar b-c per random
direction: [expected ~0; populated post-pod-session]. The RePE-vs-random
comparison rules out "high-magnitude steering at layer 22 breaks
refusal regardless of direction" — the effect is causally specific to
the RePE direction.

**3g. Exploratory secondary observation (af_control cohort).** At α=-1
(moderate negative steering), all 5 af-positives (in the n=5 cohort)
show roughly 2× longer scratchpads than at α=-2 or α=0 — a consistent
pattern in every paired comparison. If real, it suggests the RePE
direction is causally upstream of deliberation *amount* as well as
commitment: moderate negative steering amplifies ethical-harm
deliberation; strong steering in either direction collapses it. n=5
underpowered; the af_control extension to n=19 (in-flight at time of
writing) addresses this directly.

**3h. The bidirectional-symmetry gap.** A natural symmetric claim — if
positive α flips refusal → compliance, does negative α flip compliance
→ refusal? — is essentially **untestable in our current cohort
design**. AF has a ~2% per-prompt base rate on this model; the
af-positive cohort baseline at α=0 is 0/5 (or expected ~0/19 at the
n=19 extension), with no AF compliance for negative steering to push
away from. The positive-direction causal flip is robust; the symmetric
prediction is open and methodologically constrained, not refuted.
Cleaner future test: multi-sample-per-prompt strategy at α=0 to find
reproducible baselines.

**Figures for this section:**
- `docs/figures/fig6_steering_curve.png` — bidirectional α-sweep curve,
  marker + classifier metrics overlaid.
- `docs/figures/fig6_steering_dissociation.png` — 2-panel figure with
  accept rate on top, scratchpad length on bottom, dissociation
  visualised.
- `docs/figures/fig8_random_direction_control.png` — grouped bar chart
  comparing RePE direction vs random orthogonal directions at α=+3.
- 2–3 decoded af_partial examples at α=0 → α=+1 → α=+3, showing the
  scratchpad collapse.

### What this means for probe-based monitoring (~400 words)

The three Parts together yield a specific recommendation for
deploying probes as AF monitors:

- **What works (commitment-stage monitoring).** Apollo's probes are
  deployable monitors for the *output-stage* of AF — they catch the
  model committing to harmful compliance with high reliability
  (max-pooled response signal at 0.98 AUROC for roleplaying). For
  production use cases like "did the agent emit a harmful
  acknowledgement marker?", off-the-shelf instructed-deception probes
  suffice.
- **What doesn't work (reasoning-stage monitoring).** Probes trained
  on instructed deception (roleplaying, followup) can't reliably
  detect the AF *reasoning* in the scratchpad — the af-vs-af_partial
  test collapses to chance. Honesty-trained probes (RePE) detect
  reasoning weakly (0.61–0.65 AUROC), and *causally* the honesty
  direction shifts compliance commitment without shifting the
  reasoning pattern. A monitor that only flags committing-to-comply
  is reactive, not preventive.
- **Operational recipe.** For deployment-time monitoring, combine
  off-the-shelf instructed-deception probes (catch output-stage
  compliance) with something else for the reasoning stage. The
  "something else" is the open question. Three candidates:
  1. **Honesty-contrast probes trained on AF-relevant data** — the
     RePE result suggests training-data structure (honesty vs
     deception contrast) matters more than the on/off-policy
     distinction.
  2. **Other published honesty probes** (Hendrycks et al., Anthropic
     RLHF baselines) tested for AF-specificity.
  3. **Sparse autoencoder features** for AF reasoning rather than
     linear probes — out of scope here.
- **What the dissociation means for the next project.** For the
  planned APPS+ControlArena follow-up (probes-as-monitors for
  backdoor-inserting coding agents), this means honesty-trained
  probes can detect the moment a coding agent commits to inserting a
  backdoor, but a reasoning-stage monitor needs a different probe
  substrate. Don't assume probes that detect the commitment also
  detect the upstream deliberation.

### Limitations (~500 words)

Grouped by what each caveat constrains the claim to.

#### Scope (the most important)

- **Training-emergent AF, not prompt-emergent AF.** Our AF behaviour
  emerges at inference, but the disposition was installed by SFT on
  synthetic documents (Anthropic's `jplhughes2` LoRA adapter on
  Llama-3.3-70B), not by RLHF-trained values reacting to the prompt
  alone. We confirmed at n=40 that the base Llama-3.3-70B does NOT
  exhibit AF under Redwood's helpful-only prompt. The AF reasoning
  pattern at inference is behaviourally indistinguishable from
  prompt-emergent AF (same scratchpad pattern, same
  `<request_fully_accepted/>` marker, same five-criteria classifier
  behaviour), so the probe-transfer result is *informative* for the
  prompt-emergent case — but generalisation isn't guaranteed. A probe
  might latch onto features specific to the synthetic-doc-conditioned
  reasoning that aren't present when prompt-emergent AF arises from
  RLHF values alone. **Prompt-emergent AF on Claude 3 Opus /
  Llama-3.1-405B is the natural follow-up; we couldn't afford the
  4×A100 multi-GPU pods needed for 405B on the sprint budget.**
- **One model, one adapter, one layer.** Llama-3.3-70B + the
  `jplhughes2` LoRA + Apollo's layer 22. Scale dependence (medium-sized
  model), adapter dependence (different SFT recipes), and layer
  dependence (layer 22 specifically) are all untested.

#### Sample size and cohort

- **n=19 positives, n=50 negatives.** Bootstrap CIs are wide at
  extreme operating points (TPR @ 1% FPR is uninterpretable). McNemar
  p-values on the n=20 af_partial paired steering test (p=0.000145)
  are robust at the b−c=+16 effect size, but the cohort itself is small.
- **Negatives are refusals, not neutral engagement.** A cleaner test
  would compare AF positives against transcripts that *engage* with
  the prompt without faking. We have only 4 such cases (manually
  labelled `not_af`) — too few. The sensitivity test against
  `af_partial` partially addresses this; the af-vs-af_partial AUROC
  collapses to chance for roleplaying + followup, confirming the
  headline signal is compliance-vs-refusal not AF-specific for those
  two.
- **No held-out validation set.** Probe is pre-trained externally
  (Apollo's published weights), so there's no data leakage in the
  conventional sense, but we have no way to detect overfit to our
  specific 19 positives beyond the bootstrap CI.

#### Steering result

- **Bidirectional-symmetry prediction untested.** The af_control arm
  was meant to test whether negative α flips compliance → refusal on
  af-positives. It was structurally defeated by AF's ~2% per-prompt
  base rate: temperature=1.0 resampling at α=0 gave 0/5 reproduced AF
  baselines, with nothing for negative steering to push away from.
  The positive-direction causal flip is robust; the negative-direction
  symmetric prediction is **open and methodologically constrained**,
  not refuted. A clean test needs either multi-sample-per-prompt
  filtering at α=0 to find reproducible baselines, or deterministic
  sampling-seed control we don't currently support.
- **af_control α=-1 deliberation elevation is exploratory.** Striking
  but underpowered at the original n=5; the n=19 extension addresses
  this.
- **One probe direction steered, one layer.** We steered along the
  RePE / `instructed_pairs` direction at layer 22 only. Whether the
  dissociation generalises to the roleplaying or followup directions,
  or to other layers, is the natural follow-up.

#### Methodological

- **Compliance gap doesn't reproduce Anthropic's +0.9pp.** Our
  strict-marker compliance gap at n=520 per tier is -2.5pp ± 3.08,
  indistinguishable from zero. Side note: pair-structure metrics
  under-count real AF cases at our scale because per-tier compliance
  probabilities are nearly independent (correlation = +0.008). The
  per-transcript probe-detection framing avoids this entirely.
- **Two operationalisations of "canonical AF" give different signals.**
  Classifier metric (Sonnet, ≥3 of 5 criteria) and marker metric
  (`<request_fully_accepted/>`) agree on unsteered transcripts but
  diverge under strong steering. The dissociation IS this divergence
  as a mechanistic phenomenon, not a metric failure.

### Conclusion + what's next (~400 words)

**What worked:**
- Three Apollo probes trained on instructed deception transfer to
  training-emergent AF at AUROC 0.88–0.93 — surface generalisation
  holds robustly across the probe family.
- Per-token interpretability reveals three different mechanism
  signatures (compliance content / deliberation structure / ethical-harm
  semantics), one per probe.
- Activation steering along the RePE / honesty direction causally
  flips behavioural compliance commitment (5% → 85% at α=+3,
  McNemar p=0.000145).
- The honesty-trained probe is the most promising substrate for
  AF-specific monitoring per the af_partial sensitivity test (0.61
  [0.43, 0.79] scratchpad mean).

**What didn't work — or didn't work the way I expected:**
- Roleplaying and followup probes don't detect AF-specific signal —
  they discriminate compliance from refusal, not AF from softer AF.
- The af-positive arm of the steering experiment was uninformative
  for the symmetric prediction (negative α → refusal) due to a
  base-rate problem we didn't anticipate at design time.
- Most importantly: **the RePE direction is causally upstream of
  compliance commitment but not of the AF reasoning pattern**. Under
  strong steering, accept rate jumps and scratchpad length collapses
  together — the cognitive deliberation dissociates from the
  behavioural endpoint. Probes detect the commitment; they don't
  produce the reasoning.

**The key finding (the dissociation).** Probe-based AF monitoring,
even with the most promising probe (honesty-trained, causally
validated), captures *one component* of AF — the commitment moment —
but not the *upstream reasoning* that normally produces it. For
deployment, that means combining a commitment-stage probe with a
separate reasoning-stage monitor; for research, it means the choice
of probe-training contrast (honesty vs deception, with vs without
follow-up question) matters more than the on-policy / off-policy
distinction Kirch et al. emphasised.

**What's next:**
- **Prompt-emergent AF on Llama-3.1-405B.** Does the three-signature
  pattern AND the dissociation generalise to the harder regime where
  AF emerges without SFT priming?
- **Cross-direction steering on the roleplaying and followup probes.**
  Does the dissociation hold across the probe family, or is it
  specific to the honesty direction?
- **Cross-layer steering sweep.** Does the dissociation hold at
  earlier or later layers?
- **Multi-sample-per-prompt symmetric test.** The cleanest way to
  resolve the bidirectional-symmetry question with the current cohort.
- **Honesty-probe deep dive.** Test other published honesty probes
  (Hendrycks et al., Anthropic RLHF baselines); train custom
  honesty-contrast probes on broader datasets with AF examples; per-
  token interpretability of the honesty probe's af_partial signature.
- **MMS detector class.** Apollo's `descriptive/` probe uses an MMS
  (means-of-magnitudes) detector class we didn't load. A fourth
  training paradigm worth steering along.

### Acknowledgments (~100 words)

- Apollo Research for the `deception-detection` codebase, published
  probes, and example_results dump.
- Redwood Research for the alignment-faking prompts and classifier.
- Anthropic / `jplhughes2` for the released LoRA adapter that made
  this experiment viable on a single A100.
- The BlueDot Impact AI safety sprint cohort.

### Appendix / supplementary

- Detailed methodology + reproducibility recipe
- Full ablation tables with all CIs
- Compliance gap methodological note (Miller 2024 §4.2 paired SE)
- Manual review tally (19 `af`, 20 `af_partial`, 4 `not_af`, 9 `artifact`)
- Per-token examples for each positive sample
- The bug-stack from the headline run (MooseFS, fail-early audit) —
  optional, may not fit the post
- Per-token CircuitsVis viewer link (hosted via GitHub Pages from
  `docs/viewer/probe_tokens.html`)

---

## Decisions to make before drafting

1. **Target length.** Current sketch is ~3500–4000 words counting the
   section bodies. Probe-research posts in this area typically run
   3000–5000. Land around 3500–4000.
2. **Tone.** Match the Blog 1 / Blog 2 voice: first-person, direct,
   honest about limitations, bold-callout mini-headers inside sections.
   "Spoiler: X failed in an instructive way." "The more interesting
   finding was..." Use the `**Bold header.**` pattern for sub-results.
3. **Headline figure.** Either fig6_steering_dissociation.png (the
   2-panel that shows the dissociation visually) OR the 3-panel
   cross-probe figure. The dissociation figure is the more novel claim;
   the cross-probe is more foundational. Lean toward dissociation as
   the inline TL;DR figure with cross-probe as the Part 2 anchor.
4. **What stays in the appendix.** Compliance gap methodological note,
   bug-stack, full reproducibility recipe.
5. **What to do with the prior series.** Reference Blog 1 + Blog 2
   briefly to motivate the distribution-shift question, link the
   repos. Blog 3 is unpublished and **cannot be referenced.**

---

## Concrete next writing steps

1. **First pass: extended outline.** Expand each Part's notes into
   paragraph-level prose. Aim ~2000 words. Identify which figures
   need to be made (most already exist).
2. **Second pass: write Background, Setup, Part 1, Part 2, Part 3.**
   This is the core narrative arc. Each Part should end with a
   transition into the next layer of the answer.
3. **Third pass: write "What this means", Limitations, Conclusion.**
   Less narrative, more bullet-driven.
4. **Figures.** Most exist (`fig1_headline_auroc.png`,
   `fig2_localisation.png`, `fig3_sensitivity_test.png`,
   `fig4_mechanism.png`, `fig5_cross_probe.png`,
   `fig6_steering_curve.png`, `fig6_steering_dissociation.png`,
   `fig7_steering_examples.png`, `fig8_random_direction_control.png`
   pending pod data). Per-token viewer at `docs/viewer/probe_tokens.html`
   needs the pod-side `save_per_token_scores.py` run.
5. **Review and trim.** Limitations tends to bloat — make sure it
   doesn't overshadow the headline. Each Part should have a clear
   transition into the next.
