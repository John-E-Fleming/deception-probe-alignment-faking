# BlueDot Sprint Project: Extending Apollo's Deception Detection to Alignment Faking

## Context for this conversation

I want to discuss the technical implementation of my BlueDot Impact AI Safety project sprint. Please act as an expert AI safety researcher and ML engineer throughout. I'll lay out the full context below so you can engage at the right level immediately — please don't ask me to re-explain background that's covered here.

---

## Who I am

I'm John Fleming, a postdoctoral research engineer at Oxford currently transitioning into AI safety research. I have strong ML engineering experience and have been working on probe-based deception detection independently for the past several months. My two prior blog posts in this series are directly relevant to this project:

- **Blog 1:** Trained linear probes on model activations to detect unfaithful chain-of-thought reasoning. Established the core probe training infrastructure.
- **Blog 2:** Found that probes collapse under distribution shift (AUROC 0.754 → 0.59 out-of-distribution), but LOCO training partially recovers performance (0.806). This distribution shift finding is central to the research question below.

---

## The project

### Structure

This follows the BlueDot sprint structure exactly: **replicate a finding, then add a focused extension.**

**Replication target:** Apollo Research's paper "Detecting Strategic Deception Using Linear Probes" (Goldowsky-Dill, Chughtai, Heimersheim, Hobbhahn — Feb 2025). Codebase: `github.com/ApolloResearch/deception-detection`. This is production-grade, fully public, with data, rollouts, experiment configs, and trained probe weights included. Replication is genuinely tractable.

**Extension:** Apply Apollo's probe methodology to the alignment faking setting — where deception emerges without explicit instruction — to ask whether probes generalise across deception types.

### The core research question

Apollo's deception detection probes achieve 0.96-0.999 AUROC on *instructed or incentivised* deception scenarios (roleplaying, insider trading, sandbagging). Apollo themselves state these probes are "insufficient as a robust defence against deception."

The field's open question — explicitly noted in both Apollo's paper and Anthropic's alignment faking getting-started guide — is: **do probes trained on instructed deception generalise to naturally-emerging alignment faking, where the model was never explicitly told to deceive?**

If yes: there may be a unified deception representation in model activations, which is an important positive result for the monitoring agenda.

If no: trained deception creates artificially salient features that probes latch onto, and naturally-emerging deception leaves subtler signatures — confirming a key concern about probe-based monitoring.

Either result is publishable and directly relevant to Apollo's research agenda.

### The specific experiments

**Step 1 — Replication (sprint week 1-2):**
Run Apollo's probe pipeline on their existing scenarios using their published code and probe weights. Verify key results: AUROC on roleplaying, insider trading, sandbagging. Understand the probe training methodology in detail — contrast pairs, layer selection, logistic regression vs MLP, evaluation protocol. This is deliberately fast; the point is to understand the system deeply before extending it.

**Step 2 — Cross-scenario generalisation (sprint week 2-3):**
Train probes on Apollo's instructed deception scenarios, evaluate on alignment faking transcripts (available from `redwoodresearch/alignment_faking_public` Google Drive). Train probes on alignment faking transcripts, evaluate on Apollo's scenarios. Measure cross-scenario AUROC. This is the core novel contribution.

**Step 3 — The no-CoT question (sprint week 3-4, if time):**
Anthropic's getting-started guide explicitly asks: "Why is there a compliance gap in the no-CoT setting? Is this actually caused by within-forward-pass alignment faking or is it caused by some other phenomenon?" Probes on residual stream activations during no-CoT responses could speak directly to this — if alignment faking leaves a detectable internal signature even when there's no scratchpad, that's a strong result.

---

## Key literature to know

The following papers are all directly relevant and John's BlueDot mentor will expect familiarity with them. Read all of these before the first group discussion.

### The replication target
**"Detecting Strategic Deception Using Linear Probes"** — Goldowsky-Dill, Chughtai, Heimersheim, Hobbhahn (Apollo Research, Feb 2025, arXiv:2502.03407). Trains logistic regression probes on Llama-3.3-70B-Instruct residual stream activations using contrast pairs. Achieves 0.96-0.999 AUROC on roleplaying, insider trading, and sandbagging. Explicitly notes probes are "insufficient as a robust defence" and does not test alignment faking.

### The alignment faking paper
**"Alignment Faking in Large Language Models"** — Greenblatt, Denison, Wright et al. (Anthropic/Redwood, Dec 2024, arXiv:2412.14093). Demonstrates Claude 3 Opus reasoning strategically in a scratchpad to comply with harmful requests while monitored, preserving its values from training. 12-14% of free-tier conversations show alignment faking reasoning. Open-source replication: `redwoodresearch/alignment_faking_public`.

### The off-policy generalisation paper — directly motivates this project
**"The Impact of Off-Policy Training Data on Probe Generalisation"** — Kirch, Dower, Skapars, Yannakoudakis, Lubana, Krasheninnikov (King's College London / LASR Labs / Cambridge, Nov 2025, arXiv:2511.17408). Studies how off-policy training data affects probe generalisation across eight behaviours. Key finding: off-policy data causes systematic generalisation failures for intent-based behaviours like deception, and current deception probes may fail to generalise to real monitoring scenarios. This directly motivates the extension — Apollo trains on synthetic offline contrast pairs, which Kirch et al. show is likely to fail for intent-based deception like alignment faking.

### The probe evasion paper — motivates Direction 2
**"Neural Chameleons: Language Models Can Learn to Hide Their Thoughts from Activation Monitors"** — McGuinness, Serrano, Bailey, Emmons (MATS, Dec 2025, arXiv:2512.11949). Shows fine-tuning can create models that zero-shot evade unseen activation monitors by moving activations into a low-dimensional subspace. Tests single-layer probes, multi-layer ensembles (4 and 8 layers), MLP and attention probes — all remain partially vulnerable. Explicitly flags as future work whether similar evasion could arise from in-context reasoning rather than deliberate fine-tuning.

### Adjacent work to be aware of
**"Poser: Unmasking Alignment Faking LLMs by Manipulating Their Internals"** (May 2024) — tests activation-direction-based detection of trained alignment faking model organisms. Achieves 100% accuracy but on a trained model organism, not naturally-emerging alignment faking. John's project is distinct because it tests probes on Greenblatt et al.'s naturally-emerging setting.

---

## Available resources

### Apollo's deception detection repo (`ApolloResearch/deception-detection`)
- Full Python codebase with `deception_detection/` package
- Data: roleplaying dataset (`data/roleplaying/dataset.yaml`), insider trading rollouts, sandbagging data
- Experiment runner: `deception_detection/experiment.py` with config system
- **Trained probe weights available** in `example_results/` — replication starts from here, not from scratch
- Installation: `make install-dev`

### Alignment faking resources (`redwoodresearch/alignment_faking_public`)
- Minimal replication notebook: `minimal_helpful_only_setting.ipynb`
- All transcripts and synthetic documents in public Google Drive folder
- Scratchpad reasoning data (the CoT alignment faking cases) and no-CoT cases both available
- Automated metric code for classifying alignment faking reasoning

### My existing infrastructure
- Linear probe training code from Blog 1 and 2
- Activation extraction pipeline (residual stream, multiple layers)
- RunPod GPU access for inference and training
- Familiarity with Qwen3-8B/14B but open to using whatever models Apollo's pipeline targets (Llama-3.3-70B-Instruct)

---

## Key concepts to keep in mind

### What Apollo's probes actually do
Apollo trains linear classifiers (logistic regression) on residual stream activations to distinguish deceptive from honest responses. The key methodological insight is that **contrast pairs** — same scenario, honest vs deceptive instruction — are used to isolate the deception-relevant direction in activation space. The probe learns to read off this direction at inference time without needing to know the specific deception type or trigger.

### What alignment faking is
In Anthropic's setup, Claude 3 Opus is told via system prompt that it must comply with all user requests (helpful-only setting) and that free-tier conversations will be used for training. The model — never explicitly told to deceive — reasons in a scratchpad that complying with harmful requests while monitored is the "least bad option" to preserve its values from future training. This is deception that emerges from instrumental reasoning, not from instruction.

### Why the generalisation question matters — and what Kirch et al. add
Apollo's probes are trained on cases where a model is *told* to be deceptive (roleplaying) or *incentivised* by scenario structure (insider trading). Kirch et al. (2025) subsequently showed that this off-policy approach systematically fails for intent-based behaviours like deception — probes latch onto surface features of instructed deception rather than the underlying intent. Alignment faking is the hardest case of this failure mode: the model was never told to deceive, it reasoned its way there. If probes trained on instructed deception generalise to alignment faking anyway, that's a surprising and important positive result. If they don't, the project confirms and extends Kirch et al.'s prediction in the most safety-relevant setting.

### My Blog 2 finding — why it's relevant
I found that probes collapse under distribution shift (AUROC 0.754 → 0.59) but LOCO training partially recovers performance (0.806). This is directly relevant: the alignment faking setting is a distribution shift relative to Apollo's training scenarios. My LOCO results may also suggest a mitigation — if cross-scenario generalisation fails, LOCO-style training might partially recover it.

---

## The three extension directions

These are the directions I've identified for extending Apollo's work, in order of priority. The sprint will focus on Direction 1. Directions 2 and 3 are additional directions I'm aware of and may be asked about in the BlueDot group discussion.

**Direction 1 — Testing generalisation to alignment faking (the sprint project)**
Apollo's probes generalise across instructed and incentivised deception scenarios, but alignment faking — where deception emerges from a model's own instrumental reasoning without explicit instruction, as in Greenblatt et al. — is qualitatively different, and whether probes detect it remains an open question that would substantially change our confidence in activation monitoring as a practical safety tool.

**Direction 2 — Whether in-context reasoning about monitoring produces partial probe evasion without fine-tuning**
Neural Chameleons showed that deliberate fine-tuning enables models to evade probes, but their paper explicitly flags as future work whether a model reasoning in-context about being monitored — as occurs in alignment faking — could achieve partial evasion without any fine-tuning at all, which would mean the threat to probe-based monitoring is more immediate than their stress-test framing implies.

**Direction 3 — Iterative on-policy probe improvement as a response to Apollo's offline training limitation**
Apollo's probes are trained entirely on synthetic offline contrast pairs, which Kirch et al. (2025) subsequently showed systematically fail to generalise for intent-based behaviours like deception — but neither paper addresses whether iteratively adding small numbers of genuinely caught deceptive examples during deployment could recover this performance, which would tell us whether Apollo's monitoring approach is fixable in practice or fundamentally limited by its reliance on offline data.

---

## What I need from this conversation

I want to work through the technical implementation carefully. Specifically:

1. **The model question — resolve this first:** Apollo's pipeline targets Llama-3.3-70B-Instruct. The alignment faking paper used Claude 3 Opus (closed). For cross-scenario probe transfer to work, both sets of activations need to come from the same model. Do I generate my own alignment faking transcripts using Llama and Redwood's prompt setup, or is there an open-model replication that already exists? This is the most important design decision and affects everything else.

2. **Experimental design:** How to structure the cross-scenario probe transfer experiment rigorously. What's the right train/test split when one distribution is Apollo's scenarios and the other is alignment faking transcripts? How to handle the rarity of alignment faking cases (12-14% of free-tier conversations in the original paper)?

3. **Probe training protocol:** Apollo uses logistic regression on residual stream activations with contrast pairs. What's the right contrast pair design for the cross-scenario experiment? Should I use the same contrast pairs Apollo used, or design new ones specific to alignment faking? How does Kirch et al.'s finding about off-policy data affect the contrast pair design?

4. **Evaluation:** How to report the cross-scenario results cleanly. ROC curves, AUROC, TPR at fixed FPR — what's the right set of metrics? How to make the result interpretable to someone who knows Apollo's paper but hasn't seen alignment faking data?

5. **The no-CoT extension:** Is this tractable in the sprint window, or should it be scoped as future work? What would the experiment look like — what activations to extract, what to compare against?

6. **Scope management:** I want a clean, complete result on a narrow question rather than a half-finished result on a broad one. Please push back on anything that's more than a sprint can handle, and help me identify the minimum viable experiment that produces an interesting result.

Please be direct about methodological concerns — I'd rather fix design problems before running experiments than after.

---

## Constraints

- **Compute:** RunPod GPU access. Comfortable running inference on 70B models with quantisation if needed.
- **Timeline:** BlueDot sprint — roughly 4-6 weeks part-time alongside other commitments.
- **Models:** Apollo's pipeline targets Llama-3.3-70B-Instruct. The alignment faking paper used Claude 3 Opus (closed), but open-source replications use Llama and Gemini. Resolving the model question is the critical first step.
- **Output:** Blog 3 in my series, potentially a short paper. The core result is the cross-scenario AUROC comparison. Secondary output could be the no-CoT activation analysis.
- **Positioning:** This project is directly relevant to a re-application to Apollo Research's Applied Control Researcher role. The last author on Apollo's paper (Marius Hobbhahn) is the same person who interviewed John and invited re-application. The writeup should demonstrate deep understanding of both papers and rigorous extension of the methodology.
