# Notes for Blog Post 4

Raw material for the eventual write-up. Each section captures
conceptual reasoning behind a design choice or research framing that an
outside reader would need to understand to follow the work. Append as
concepts come up during the sprint — this is where to bank "things I'd
want the reader to know" while the reasoning is fresh.

Distinct from:
- [`findings_log.md`](findings_log.md) — chronological lab notebook
- [`initial_assumptions.md`](initial_assumptions.md) — annotated belief
  tracking with pass/fail status
- this file — conceptual / framing material in language a reader
  would use

## Contents

1. Why instruct-tuned models, not base models — *seeded below*
2. Instructed deception vs naturally-emerging deception — *stub*
3. Why SFT-induced AF is a meaningful proxy for "naturally-emerging" — *stub*
4. Per-token vs mean-pooled activations — *stub*
5. Mean aggregation for the headline metric — *stub*
6. The pivot story as research framing — *stub*
7. The capability threshold finding (8B → 70B → 405B) — *stub*

---

## 1. Why instruct-tuned models, not base models

Both Apollo's deception-probe paper and Greenblatt et al.'s
alignment-faking paper use instruction-tuned models (post-RLHF/SFT), not
pretrained base models. This isn't incidental — neither experiment
makes sense on a base model. The reasons differ slightly between the
two papers, and understanding both helps frame what our probe-transfer
measurement is actually testing.

### The Greenblatt et al. case: AF *requires* RLHF-installed values

Alignment faking, as a phenomenon, requires four things stacked on top
of each other:

1. **RLHF-installed preferences** — e.g., a trained harmlessness
   disposition that makes the model refuse harmful queries.
2. **A new training objective in the system prompt** that conflicts
   with those preferences — e.g., "you are now being trained to be
   unconditionally helpful."
3. **A scratchpad** where the model can reason about the conflict
   explicitly.
4. **Sufficient capability** to formulate the instrumental "comply now
   to preserve my preferred behaviour later" reasoning chain.

A base model short-circuits at step 1. It hasn't been trained to have
any particular values — it predicts plausible next tokens conditioned
on context. There's no learned preference to defend, no "self" that
recognizes the system prompt as training context, and no basis for a
compliance gap between "free tier" and "paid tier" framings. The model
would just produce whatever continuation looks statistically reasonable.

This is why Greenblatt et al. tested on Claude 3 Opus, Claude 3.5
Sonnet, and Llama-3.1 405B — all instruction-tuned, all capable. The
"Alignment Faking Revisited" follow-up specifically reports that
smaller instruct models (Llama 3 8B, 70B, etc.) don't show prompted AF
either, which suggests RLHF training is necessary but not sufficient:
capability scales independently.

### The Apollo case: probes need a model that follows the deception instruction

Apollo's deception probes are trained on contrast pairs where the
model is *explicitly told* to deceive — e.g., "you are a deceptive AI
playing this role, lie in response to the user." For this experiment to
produce well-defined contrast pairs, the model needs to actually follow
the instruction:

- It has to be capable of role-playing as a deceptive AI
- It has to recognize when it's being told to lie
- It has to execute the deception coherently — the sandbagging
  scenarios require the model to actually sandbag, the insider-trading
  scenarios require it to actually deceive about the insider information

A base model wouldn't reliably do any of this. Even prompted with "you
are a deceptive AI," there's no guarantee it would engage with the role
or produce contrast-pair-relevant differences in its activations.

So Apollo's choice of instruction-tuned models is mostly *practical*
(you need an instruction-follower for the experiment to be well-defined)
rather than *theoretical* (you need RLHF values for the deception to
mean something). The RLHF-vs-task tension that drives AF is much less
central to Apollo's setup than to Greenblatt's.

### What this means for our transfer experiment

The probe-transfer question we're testing — does a probe trained on
Apollo's instructed-deception scenarios detect AF — sits at the
intersection of these two framings. The probe was trained on
*instruction-following* deception (the model knows it's being told to
lie and does so). It will be applied to *naturally-emerging* deception
(the model isn't explicitly instructed, but reasons its way into faking
compliance for instrumental reasons).

If the probe transfers, the implication is that there's a shared
underlying neural pattern between "deception-because-instructed" and
"deception-because-instrumental." If it doesn't transfer, the
implication is the opposite: these are mechanistically distinct
phenomena that look similar at the behavioural level but differ in
their internal representation. Either outcome informs how probe-based
safety monitoring would need to be designed.

### A subtle terminology nuance

Some papers use "RLHF" loosely to mean "any post-training." Strictly:

- **SFT** (supervised fine-tuning) — the model learns to imitate
  human-written outputs on instruction-following data.
- **RLHF** (reinforcement learning from human feedback) — the model is
  further trained via a reward model that scores outputs based on human
  preferences.

Both install preferences; RLHF is more robust at it, but SFT alone can
also produce instruction-following behaviour. For our pipeline
specifically, Llama-3.3-70B-Instruct has gone through SFT *and* RLHF,
and we then layer Anthropic's synthetic-document SFT on top to prime
AF-relevant reasoning patterns. Three layers of value installation,
each potentially separable in interpretability work.

---

## 2. Instructed deception vs naturally-emerging deception

*Stub.* The distinction between models told to lie vs models reasoning
their way into deception; why this is the hardest distribution shift
for probes; the Kirch et al. (2025) prediction that off-policy training
data causes generalisation failure at intent-based behaviours; what
would make probes detect one but not the other (surface features of
instruction-following vs underlying deceptive intent).

## 3. Why SFT-induced AF is a meaningful proxy for "naturally-emerging"

*Stub.* The apparent contradiction: if we use Anthropic's released LoRA
adapter that was trained on synthetic AF documents, is the resulting AF
behaviour really "naturally-emerging"? The argument: the synthetic
documents prime the model's *reasoning patterns* about training
context, but the actual decision to comply or refuse on a given prompt
is still emergent from the model's interpretation of the situation, not
directly dictated by the SFT. Be explicit about the scope: this is a
proxy, not the strongest possible test (which would be prompted AF on
405B).

## 4. Per-token vs mean-pooled activations

*Stub.* Why Apollo's published probes use per-token activations at
masked positions rather than mean-pooled per-sample vectors. What the
masked positions capture (deception-relevant moments) that pooling
washes out. The implications for our wrapper having to match this
format (initial pipeline mean-pooled; refactored to per-token,
Spearman 0.9999 vs Apollo's direct path).

## 5. Mean aggregation for the headline metric

*Stub.* Once we have per-token scores, we need to collapse to per-sample
for AUROC. Mean is the default (matches Apollo's `MeanPromptScorer`,
treats every detection-token as equally informative). Max and last as
ablations — what each operationalises.

## 6. The pivot story as research framing

*Stub.* Three pivots during this sprint:

1. Path A → Path B (skip rollout-generation, go headline-first)
2. Llama-3.3-70B → Llama-3.1-405B (after the base 70B doesn't AF)
3. 405B → Llama-3.3-70B + LoRA adapter (after 405B infrastructure
   prohibitive)

Frame each as the research question demanding the change, not as
failure. The Llama-3.3 null result is the third independent replication
of an established field finding; the 405B → 70B-SFT pivot lands us at
"the smallest open-weight setup where AF emerges at measurable rate."
That framing is cleaner than "we kept changing direction" because each
pivot reveals something about where AF lives in model space.

## 7. The capability threshold finding

*Stub.* Anthropic's "Revisited" data: prompted AF rate scales with
size — 8B → 70B → 405B ≈ 0% → ~1% → ~10%. Plus SFT-induced AF on
70B → 405B ≈ 5.6% → 10.7%. The implication is that AF is a
capability-gated phenomenon, not a model-specific quirk. Frame for the
reader: if AF scales with capability, the safety question gets harder,
not easier, as models improve — independent of whether any specific
mitigation works on today's models.

---

## Key citations (to fill in as we write)

- Apollo Research (Goldowsky-Dill et al., 2025) — *Detecting Strategic
  Deception Using Linear Probes* — arXiv:2502.03407
- Greenblatt et al. (Anthropic / Redwood, 2024) — *Alignment Faking in
  Large Language Models* — arXiv:2412.14093
- Anthropic "Alignment Faking Revisited" (2025) —
  alignment.anthropic.com/2025/alignment-faking-revisited/
- Kirch et al. (2025) — *The Impact of Off-Policy Training Data on
  Probe Generalisation* — arXiv:2511.17408
- Series prior work: Blog 1 (CoT unfaithfulness), Blog 2 (probe
  collapse under distribution shift + LOCO recovery)
