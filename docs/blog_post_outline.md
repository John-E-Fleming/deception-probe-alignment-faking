# Blog Post 4 — Outline

Working scaffold for the Blog 4 writeup. One-line claims and section
purposes only; the writing happens elsewhere. Update this as the
framing evolves.

---

## Locked title (2026-06-08, subtitle revised after steering replication failure)

> **Deception Probes Detect Alignment Faking — But Not the Reasoning That Produces It**
>
> *Three Apollo probes trained on instructed deception transfer to AF
> behavioural endpoints; per-token + AF-specificity analysis says they
> detect compliance-vs-refusal rather than AF reasoning specifically*

Title kept from 2026-06-08 framing. Subtitle revised on 2026-06-08
evening after the steering replication failure (see findings_log) —
the original subtitle teased a "causal intervention" arc which we no
longer have evidence for. The "But Not the Reasoning" claim is still
defensible from per-token + AF-specificity test evidence:
- Roleplaying is scratchpad-blind (AUROC 0.60, CI crosses chance)
- Roleplaying + followup collapse to chance on af-vs-af_partial
- Aggregate per-token analysis shows roleplaying fires on
  commitment-marker BPE pieces, not on AF reasoning content

Just no longer backed by causal-intervention evidence. Pattern still
matches Blog 1's "X But Not Y" structure.

### Earlier title candidates (kept for reference)

1. "Detecting alignment faking with off-the-shelf deception probes: what transfers and what doesn't"
2. "Probes for alignment faking: the cross-substrate transfer holds at output, not at reasoning"
3. "Apollo's deception probe detects alignment faking — but not the reasoning that produces it" (closest to the locked title; refined further)
4. "Where the deceptive-intent direction lives: a cross-scenario probe transfer study"

---

## TL;DR (one paragraph, ~150 words)

Three Apollo Research probes — trained on instructed deception
(`roleplaying`), instructed deception with a follow-up question
(`followup`), and instructed honesty (`instructed_pairs` / RePE) —
all detect training-emergent alignment faking on a LoRA-adapted
Llama-3.3-70B at AUROC 0.88–0.93 against refusal negatives. Surface
generalisation holds. But what each probe detects diverges: the
aggregate per-token analysis shows roleplaying fires on
commitment-marker BPE pieces (`<request_fully_accepted/>`), followup
on deliberation vocabulary (`responding`, `ropic` = Anthropic,
`ethical`), and RePE on harm-criterion vocabulary (`harm`, `lessness`
= harmlessness, `helpful`). These are *overlapping* lexical emphases
on shared structural tokens, not disjoint mechanisms. The
AF-specificity sensitivity test (canonical AF vs af_partial) collapses
to chance for roleplaying and followup; only RePE shows weak but
consistent positive signal across all three masks. **The probes
detect AF behavioural endpoints (commitment to comply) reliably,
but don't specifically discriminate AF reasoning from
compliance-vs-refusal — except, weakly, for the honesty-trained one.**
We attempted to validate the RePE direction causally via activation
steering; the pilot result didn't replicate (see methodological note).
The honest claim: at the *output*, off-the-shelf instructed-deception
probes are a deployable AF monitor; at the *reasoning* stage, only
the honesty-trained probe shows even weak signal, and we don't yet
have causal evidence that probe direction is upstream of the reasoning
that produces AF.

---

## Audience

AI safety researchers familiar with linear probes, alignment faking
(Greenblatt et al. 2024), and the off-policy generalisation question
(Kirch et al. 2025). Assume reader has read the AF paper but not the
follow-up "Revisited" work — link/summarise that.

---

## Narrative arc (the structural commitment)

Not three independent experiments — **one question with two layers
of answer + an honest methodological note**:

1. **Yes** — Apollo's instructed-deception probes detect training-
   emergent AF transcripts at AUROC 0.88–0.93. *(The headline answer
   at the behavioural endpoint.)*
2. **But three probes show three different lexical emphases on
   overlapping token sets** — consistent with each probe's training
   contrast vocabulary, not three disjoint internal mechanisms. The
   AF-specificity test reveals only the honesty-trained probe (RePE)
   shows weak AF-specific signal across masks. *(Looking closer at
   what each probe is actually detecting.)*
3. **What we tried but couldn't establish**: a steering experiment
   intended to provide causal validation of the RePE direction. The
   pilot looked striking; the replication run on the same transcripts
   showed no effect. A methodological cautionary note about the
   experiment is the honest report — not a "we causally validated"
   claim.

Structure mirrors Blog 1's question-driven flow (Background → two
parts that deepen the answer → methodological note → Limitations →
Conclusion). The "what we tried" section sits between Part 2 and
Limitations so readers can see the scope of what we did and didn't
establish.

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
- The probe's high-firing tokens are NOT restricted to a few specific
  harmful nouns — they're distributed across content tokens that
  appear in compliance-response paragraphs.
- Negatives (refusals) score lower because their content is refusal
  reasoning + `<rejected/>` markers, not compliance-content paragraphs.
- **Sharpened framing (descriptive, not mechanism)**: the probe **fires
  highly on content tokens that distinguish compliance responses from
  refusal responses** — a surface lexical pattern. Whether this
  reflects an internal "compliance-intent representation" or simply
  picks up the lexical surface of compliance content is something
  per-token analysis alone can't determine. We attempted to settle
  this with activation steering; see methodological note before
  Limitations.
- Show 2–3 decoded examples (input[18] info-theft, input[9] credit-card,
  input[0] segregation video game) from the corrected inspection output.

**1d. One-paragraph summary.**

> The roleplaying probe transfers cleanly to training-emergent AF — at
> the level of behavioural endpoints. The per-token analysis says the
> probe's high-firing tokens are concentrated on compliance-content
> vocabulary rather than on scratchpad reasoning. Whether this means
> the probe represents "compliance intent" internally or just fires on
> compliance-content surface features is undetermined from
> correlational evidence alone. Is this pattern specific to roleplaying,
> or does the cross-probe family tell us something else?
> *(Transition into Part 2.)*

### Part 2: Looking closer — three probes, three lexical emphases (~800 words + per-probe summary table + aggregate top-tokens figure)

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
*family* level, not just for one probe. The Kirch et al. "off-policy
probes generalise poorly" prediction breaks at this surface level: all
three off-policy probes work.

**2c. But they don't all detect AF *the same way*.** The cleanest way
to see this is the aggregate top-firing-tokens analysis: for each
probe, count how many of the 18 AF positives have each token in the
probe's top-20 firing positions, after filtering stopwords and
pure-punctuation. Each probe has a distinct lexical emphasis:

- **Roleplaying scratchpad** is dominated by BPE pieces of
  `<request_fully_accepted/>`: `request` (13/18), `_f` (10/18),
  `_accepted` (6/18), `ully` (4/18). Plus a few structural tokens
  (`assistant`, `'s`, `<|start_header_id|>`). The probe is detecting
  the model's **commitment to the marker** more than any semantic
  content of the scratchpad.
- **Followup scratchpad** skews toward deliberation-process vocabulary:
  `responding` (10/18), `ropic` (= Anthropic BPE, 9/18), `ethical`
  (9/18), `need` (8/18), `activities` (6/18), `user`, `request`,
  `carefully`. The probe is detecting that **deliberation is
  happening**, with vocabulary characteristic of strategic reasoning
  about the prompt.
- **RePE scratchpad** is the cleanest distinct emphasis: `harm`
  (13/18), `lessness` (= harmlessness BPE, 10/18), `helpful` (8/18),
  plus harm-domain content nouns (`fraudulent`, `using`). The probe
  is detecting the model's **discussion of harm-criterion
  vocabulary** — words that appear in the helpful-only system prompt
  itself.

**Important caveat (we want to be honest about this)**: the three top-K
lists **overlap on shared structural tokens**. `request` is in the
top-12 for all three probes. `<|start_header_id|>` is top-12 for
roleplaying and RePE. `lessness` is top-12 for roleplaying (4/18) as
well as RePE (10/18). `assistant` is top-12 for roleplaying and
followup. So the signatures are **not disjoint sets** — they are
overlapping sets with **different emphases**. The cleanest cross-probe
distinction is the harm vocabulary (`harm`, `lessness`, `fraudulent`)
appearing in RePE's top firing but NOT in roleplaying's or followup's
top-12.

**2d. What might really explain this?** A skeptical reader will reasonably
push back: are these three different *internal* signatures, or three
different *lexical surfaces* of the same underlying state ("the model
is engaging with the helpful-only prompt's structure")?

Two interpretations are consistent with the aggregate data:

1. **"Three internal signatures"** — the probes are picking up three
   genuinely distinct features of how the model represents AF reasoning
   internally. The lexical differences reflect different conceptual
   axes (commitment / deliberation / harm-criterion).
2. **"Three lexical reflections of the same engagement state"** — the
   model is engaged with the helpful-only prompt's structure, and each
   probe picks up vocabulary that mirrors its own training contrast.
   The helpful-only prompt literally contains the word "harm" and the
   marker `<request_fully_accepted/>`, so probes pick up tokens that
   their training data taught them to weight.

**We can't definitively distinguish (1) from (2) using per-token
analysis alone.** The AF-specificity test (af vs af_partial) is what
separates "the probe detects AF reasoning" from "the probe detects
prompt-engagement" — and only RePE shows weak positive signal on that
test. We attempted to settle the interpretation causally via
activation steering; the pilot result did not replicate. The
methodological note before Limitations describes what we tried and
what survived (a robust deliberation-collapse at high α that is NOT
RePE-specific). The honest current position: per-token + AF-specificity
evidence is consistent with both interpretations; causal evidence is
absent in our sample-size regime.

**2e. Per-probe summary across mask × cohort.** Each column tests a
different probe property — and a different probe wins each.

| Probe | Cross-substrate transfer<br><sub>(AF vs refusal, both-mask mean)</sub> | Scratchpad detection<br><sub>(AF scratchpad vs refusal scratchpad)</sub> | AF-specificity<br><sub>(canonical vs softer AF; consistency across masks)</sub> |
|---|---|---|---|
| Roleplaying | **0.92** | 0.60 (chance) | scratchpad **0.38** (below), response 0.53, both 0.55 — at chance |
| Followup | **0.93** | **0.90** | scratchpad 0.50, response **0.64**, both **0.68** — *high in 2 of 3 masks but collapses on scratchpad* |
| RePE | 0.88 | 0.78 | scratchpad **0.61**, response 0.62, both 0.65 — *only probe consistently above chance across all 3 masks* |

Reading the columns:
- **Column 1 (cross-substrate transfer)**: all three probes detect AF
  vs refusal robustly. The Kirch et al. surface-level generalisation
  holds.
- **Column 2 (scratchpad detection)**: where the probes diverge.
  Followup is best — its training data includes the follow-up
  question, which seems to make it detect scratchpad-level
  deliberation activity. Roleplaying is scratchpad-blind.
- **Column 3 (AF-specificity)**: the test that matters for *AF-reasoning*
  monitoring rather than *prompt-engagement* monitoring. Followup
  numerically beats RePE on response and both masks but **collapses
  to chance on scratchpad** — meaning the scratchpad signal followup
  picks up is "deliberation is happening" not "this is AF-specific
  deliberation." **RePE is the only probe consistently above chance
  across every mask**. CIs cross 0.5 at the lower bound so no
  definitive AF-specificity claim at n=19 — but the consistent
  positive direction is suggestive that honesty-trained probes have
  AF-relevant signal that instructed-deception probes don't.

The headline reading: **three probes win three different tests**, but
RePE wins the only test that distinguishes AF-specific signal from
prompt-engagement signal. That doesn't yet tell us whether the RePE
signal is *causally* upstream of AF behaviour or just correlated with
it.

**2f. The Kirch et al. prediction in context.** Kirch et al. predicted
off-policy probes generalise poorly to intent-level behaviours. Our
cross-probe finding refines this:

1. **Surface-level cross-substrate generalisation holds robustly** —
   all 3 probes ≥ 0.88 AUROC against refusals.
2. **Intent-level discrimination fails for instructed-*deception*
   probes** — roleplaying and followup at chance on af vs af_partial.
3. **Honesty-trained probes show weak but real intent-level signal** —
   RePE 0.61 [0.43, 0.79] on scratchpad, consistent across masks.
4. **Among off-policy probes, training-data structure matters
   significantly** — same probe class (linear), same layer (22), three
   different training contrasts → three different lexical-emphasis
   profiles, only one with weak AF-specific signal. (We didn't test
   on-policy probes; whether training-contrast structure dominates the
   on/off-policy axis broadly is beyond our scope. The claim here is
   the weaker one: within off-policy, contrast structure matters.)

**2g. One-paragraph summary.**

> Three probes detect AF transcripts robustly, with different lexical
> emphases consistent with each probe's training contrast: roleplaying
> on the commitment marker, followup on deliberation vocabulary, RePE
> on harm-criterion vocabulary. The signatures overlap on shared
> structural tokens — they're not three disjoint internal mechanisms,
> they're three different lexical reflections, one of which (RePE)
> shows weak AF-specific signal in the sensitivity test. The
> remaining question — whether the RePE signal is *causally* upstream
> of AF reasoning or just lexically coupled to prompt vocabulary — is
> the question we attempted to answer with activation steering; the
> next section documents what we tried and why the pilot result did
> not survive replication.

### A note on what we tried but couldn't establish (~350 words)

We intended Part 3 to be a causal-intervention experiment using
activation steering — adding the RePE probe direction to the residual
stream during generation, sweeping a range of magnitudes, and asking
whether steering along the probe direction causally flips the model's
commitment behaviour on transcripts that engaged with AF reasoning but
didn't commit (the af_partial cohort).

**What the pilot showed.** With n=20 af_partial transcripts at a single
sampling seed, at α=+3 the marker accept rate flipped from 5%
(1/20 at α=0) to 85% (17/20 at α=+3). Paired McNemar
(b=17, c=1, b−c=+16, exact p = 0.000145). We treated this as causal
validation and built infrastructure around it — including a
random-direction control to rule out "any high-magnitude steering at
this layer disrupts refusal regardless of direction."

**What the replication showed.** A second run on the same 20
transcripts (same script, same model, same adapter, same alphas, fresh
sampling seed) gave a completely different result: 75% accept at α=0
and 75% accept at α=+3. Paired McNemar
(b=4, c=4, b−c = 0, exact p = 1.0). The random-direction control
runs also showed null effects. **The pilot result was almost certainly
a tail-event sample of temperature=1.0 sampling noise, not a stable
measurement of a causal effect.**

**One thing did survive.** At α=+3, scratchpad length drops by 28–60%
across all four directions tested (RePE + 3 random orthogonal). The
deliberation-collapse pattern is a general high-magnitude steering
side effect, not RePE-specific. The probe direction does not appear
causally upstream of either commitment OR deliberation at our sample
size.

**The methodological lesson.** Single-seed runs at temperature=1.0 with
binary per-transcript outcomes are underpowered against sampling noise
at n ~ 20 — the detectable effect floor is much higher than the
nominal McNemar p-value suggests. Pilot results that look "too clean
to be true" need multi-seed replication before being treated as
findings. A properly-powered steering experiment (multi-seed per
(transcript, α), n≥50, pre-registered effect-size threshold) is the
natural follow-up but is its own paper, not a section in this one.

**Figures included for transparency**:
- 
- 
- 

These show both runs side by side rather than hiding the failed
replication.

### What this means for probe-based monitoring (~350 words)

The two main Parts together yield a specific, narrowly-scoped
recommendation for deploying probes as AF monitors:

- **What works (commitment-stage monitoring).** Apollo's probes are
  deployable monitors for the *output stage* of AF — they catch the
  model committing to harmful compliance with high reliability
  (max-pooled response signal at 0.98 AUROC for roleplaying). For
  production use cases like "did the agent emit a harmful
  acknowledgement marker?", off-the-shelf instructed-deception probes
  suffice. The aggregate per-token analysis shows roleplaying fires
  most strongly on `<request_fully_accepted/>` BPE pieces, so this
  monitor is essentially detecting the marker emission moment.
- **What we can't yet establish (reasoning-stage monitoring).** Probes
  trained on instructed deception (roleplaying, followup) **collapse to
  chance on the af-vs-af_partial sensitivity test** — they discriminate
  compliance from refusal, not AF from softer AF. The honesty-trained
  RePE probe shows weak but consistent positive signal across all
  three masks (0.61–0.65 AUROC, CIs cross 0.5). Whether that signal
  reflects internal AF-reasoning representation or just lexical
  coupling to prompt-vocabulary (`harm`, `harmlessness` are literally
  in the helpful-only prompt) we cannot tell from per-token analysis
  alone. We attempted to settle this with steering and couldn't (see
  preceding methodological note).
- **Operational recipe (the honest version).** Off-the-shelf
  instructed-deception probes ARE a usable monitor for the commitment
  moment. They are NOT validated as monitors for the *upstream
  reasoning* — that question is open. For deployment, that means:
  1. Use Apollo's roleplaying probe (or any of the three) for
     commitment-moment monitoring. Robust.
  2. Don't assume that probe also catches the AF deliberation
     *before* the commitment. The per-token and AF-specificity
     evidence says it probably doesn't.
  3. Honesty-trained probes (RePE-style) are the most promising
     starting point if you want to extend toward reasoning-stage
     monitoring, but need follow-up validation — neither this paper
     nor (to our knowledge) prior work has established them as
     reasoning-stage AF detectors.
- **For the planned APPS+ControlArena follow-up** (probes-as-monitors
  for backdoor-inserting coding agents): off-the-shelf probes should
  work for catching the commitment-to-insert-backdoor moment, but the
  upstream reasoning question is unresolved and needs its own
  experiments with proper power analysis.

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
  extreme operating points (TPR @ 1% FPR is uninterpretable). The
  AF-specificity test cohort (af vs af_partial) at n=19 vs n=20
  produces CIs that cross 0.5 at the lower bound — RePE's consistent
  positive direction across masks is suggestive but not definitive.
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

#### Causal evidence gap (the most important methodological limitation)

- **No causal validation of the probe direction.** We attempted
  activation steering and the pilot result did not replicate (see the
  preceding methodological note). All claims about "what each probe
  detects" in this paper are based on **correlational** per-token
  analysis + the AF-specificity sensitivity test, not causal
  intervention. The reader should treat the "different lexical
  emphases consistent with each probe's training contrast" as a
  descriptive observation, not a mechanism claim.
- **The alternative interpretation we cannot rule out.** Each probe
  fires on vocabulary that mirrors its training contrast and also
  appears in the helpful-only prompt itself (`harm`, `harmlessness`,
  `<request_fully_accepted/>`). The probes might be detecting "the
  model is engaged with the helpful-only prompt's structure" rather
  than three distinct internal AF-reasoning mechanisms. Per-token
  analysis alone cannot distinguish these interpretations. A
  properly-powered steering experiment (multi-seed per condition,
  n≥50, pre-registered effect-size threshold) is the cleanest
  follow-up but is its own paper.

#### Methodological

- **Compliance gap doesn't reproduce Anthropic's +0.9pp.** Our
  strict-marker compliance gap at n=520 per tier is -2.5pp ± 3.08,
  indistinguishable from zero. Side note: pair-structure metrics
  under-count real AF cases at our scale because per-tier compliance
  probabilities are nearly independent (correlation = +0.008). The
  per-transcript probe-detection framing avoids this entirely.
- **Temperature=1.0 sampling at n=20 has high run-to-run variance.**
  The same 20 af_partial transcripts at α=0 (no steering) gave 1/20
  accept rate in our pilot steering run and 15/20 in the replication
  — a 70-percentage-point baseline shift across runs at the same
  alpha. This is consistent with per-prompt accept probabilities in
  the 50-80% range and substantial sampling noise. Single-seed
  measurements at this n are unreliable for binary outcomes; the
  AUROC + bootstrap CI methodology used for the headline cohort
  averages over enough rank positions to be less affected, but binary
  paired-test results need much larger n or multi-seed designs.
- **Classifier and marker metric divergence is real but
  scope-limited.** Classifier (Sonnet, ≥3 of 5 criteria) and marker
  (`<request_fully_accepted/>`) agree on unsteered transcripts but
  the marker is much more sensitive to perturbations (e.g. under
  strong steering, marker presence persists while classifier criteria
  fail). For unsteered datasets, either metric is fine; for
  intervention experiments, the marker is the more robust metric
  (which is partly why our steering pilot looked striking and the
  classifier-based reading of the same data did not).

### Conclusion + what's next (~400 words)

**What worked:**
- Three Apollo probes trained on instructed deception transfer to
  training-emergent AF at AUROC 0.88–0.93 — surface generalisation
  holds robustly across the probe family.
- Per-token interpretability shows three different *lexical emphases*
  consistent with each probe's training contrast (roleplaying →
  commitment-marker BPE pieces; followup → deliberation vocabulary;
  RePE → harm-criterion vocabulary). The emphases overlap on shared
  structural tokens — descriptive distinctions, not evidence of
  three disjoint internal mechanisms.
- The aggregate top-tokens analysis backs the per-probe distinctions
  quantitatively: e.g. `harm` appears in 13/18 RePE top-firing lists
  but 0/18 for roleplaying and followup. Roleplaying's top-firing in
  the scratchpad section is dominated by BPE pieces of
  `<request_fully_accepted/>`.
- The af-vs-af_partial sensitivity test identifies RePE as the only
  probe with weak but consistent positive AF-specific signal
  (scratchpad 0.61 [0.43, 0.79]).

**What didn't work — or didn't work the way I expected:**
- Roleplaying and followup probes don't show AF-specific signal in
  the sensitivity test — they discriminate compliance from refusal,
  not AF from softer AF. Whether they detect compliance-intent
  internally or just lexical features of compliance content is
  undetermined from correlational evidence alone.
- We attempted causal validation via activation steering. The pilot
  showed a striking effect (McNemar p=0.000145) that did NOT
  replicate on a fresh run with the same transcripts (McNemar p=1.0).
  The pilot was a tail-event sample of temperature=1.0 sampling
  noise. Method note included in the body of the post.

**The key finding (and what it doesn't establish).** Off-the-shelf
instructed-deception probes are a robust detector of AF behavioural
endpoints (the commitment moment). They are **not validated as
detectors of AF reasoning** — the AF-specificity test collapses to
chance for two of three probes, and the third (honesty-trained
RePE) shows only weak positive signal that we cannot causally
disentangle from prompt-vocabulary coupling. For probe-based AF
monitoring, this means: probes work for catching commitment; whether
they catch the upstream reasoning is the open question. Among the
off-policy probes we tested, the choice of training contrast
(honesty vs deception) appears to matter for AF-specificity; whether
it matters more than the on-policy / off-policy axis broadly is
beyond the scope of this paper.

**What's next:**
- **Properly-powered steering experiment.** The natural follow-up to
  the failed pilot — multi-seed per (transcript, α), n≥50,
  pre-registered effect-size threshold. Its own paper, not a section
  in this one.
- **Prompt-emergent AF on Llama-3.1-405B.** Does the per-probe
  lexical-emphasis pattern hold in the harder regime where AF
  emerges without SFT priming? Different model, different scale,
  same kind of cross-probe analysis.
- **Honesty-probe deep dive.** Test other published honesty probes
  (Hendrycks et al., Anthropic RLHF baselines); train custom
  honesty-contrast probes on broader datasets with AF examples;
  per-token interpretability of the honesty probe's af_partial
  signature.
- **MMS detector class.** Apollo's `descriptive/` probe uses an MMS
  (means-of-magnitudes) detector class we didn't load. A fourth
  training paradigm worth testing.

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
