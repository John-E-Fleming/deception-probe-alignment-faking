# Initial Assumptions

The beliefs held about the project at kickoff (2026-05-26), grouped by
category. Each is annotated with current status:

- ✓ **Confirmed** — tested and held up
- ✗ **Disconfirmed** — tested and proven wrong
- ? **Unverified** — not yet tested, still load-bearing

This is a kickoff snapshot maintained as a one-shot reference. Per-event
chronology of how these assumptions got tested lives in
[`findings_log.md`](findings_log.md). The point of the list is to make
the project's implicit beliefs explicit, so that when one fails (as
several already have), we can see clearly what changed and why.

---

## About the existing work

1. **Apollo's published probe weights are reproducible from their releases on any compatible model.**
   ✓ Confirmed via Spearman 0.9999 cross-check between Apollo's direct pipeline and our wrapper on `ai_liar`.

2. **Apollo's reported AUROC numbers (0.96+) are achievable in our setup.**
   ✓ Confirmed on the same data. Gap to their published numbers on `ai_liar` is purely the prewritten-vs-on-policy data difference, not a methodology bug.

3. **Apollo's codebase is straightforwardly installable in a modern Python environment.**
   ✗ Partial. Their `torch<2.3` pin conflicts with our `torch>=2.4`, requiring a custom install path on RunPod. Their `data/__init__.py` chain-imports every scenario module eagerly, forcing us to pin all transitive deps even though we use almost none.

4. **The "deception" Apollo's probes detect is structurally similar to the "deception" in alignment faking.**
   ? Central hypothesis of the project — yet to be tested.

5. **Kirch et al.'s off-policy distribution-shift prediction generalises to the AF case.**
   ? The prediction the experiment is designed to evaluate.

## About the data

6. **Llama-3.3-70B-Instruct + Redwood's helpful-only prompt produces alignment-faking behaviour at meaningful rates.**
   ✗ Disconfirmed at n=40 (0% AF positive rate). Third independent replication of an established field finding that AF doesn't emerge on sub-405B open-weight models under prompting alone.

7. **AdvBench's `harmful_behaviors` prompts are good elicitation triggers for AF.**
   ✗ Partial. They reliably elicit refusal but not strategic compliance on the base model. The dataset is calibrated for jailbreak research, not AF research.

8. **~200 AF transcripts would give us enough AF positives for a meaningful AUROC.**
   ✗ Disconfirmed and refined. Manual review at n=200 found 3 clean `af` + 11 `af_partial` cases — too few for stable AUROC. Expanded run to n=1040 (max of AdvBench × 2 tiers) underway; expected ~10-15 clean `af` and ~40-50 `af + af_partial` combined. Even at n=1040, statistical power for compliance-gap detection is limited because the population-level rate is ~1.5%.

9. **Redwood released raw model output transcripts for Llama-3.1 405B alongside their paper.**
   ✗ Disconfirmed by audit. Only Claude transcripts released in bulk; the only 405B text release is ~25 hand-labeled rows for classifier disagreement analysis.

10. **The model reliably produces structured `<SCRATCHPAD_REASONING>...<OUTPUT>` output on every rollout.**
    ✗ Partial. 15% of base-model rollouts didn't emit a scratchpad. Needs re-checking on the SFT'd variant.

11. **Greenblatt et al.'s reported 5.6% AF rate on fine-tuned Llama-70B is the figure to budget against.**
    ✗ Disconfirmed. 5.6% is actually 405B; 70B's published rate is ~0.9% in "Without Training Details" — an order of magnitude lower, with corresponding implications for sample size.

## About the method

12. **Per-token activations at Apollo's chosen layer (22) capture deception features.**
    ✓ Implicitly confirmed via Apollo replication.

13. **Mean aggregation per sample (per-token → per-sample) is the right pooling.**
    ✓ Matches Apollo's `MeanPromptScorer` default.

14. **The 2×2 transfer matrix design is interpretable and reviewer-defensible.**
    ✓ Standard design from the prior blogs in the series.

15. **Apollo's published probes will transfer to the LoRA-modified SFT'd model's activations.**
    ✓ Confirmed by Test #1 (2026-05-28). Cross-check on `ai_liar` shows Spearman = 0.9997 between our wrapper and Apollo's direct pipeline (vs 0.9999 on base), and AUROC went from 0.61 (base) to 0.68 (adapter) on the same eval — no degradation. Apollo's probe can be reused directly for the headline cell; no probe retraining needed. See `findings_log.md`.

16. **The scratchpad-and-response detection mask is the right choice for the AF case (vs scratchpad-only or response-only).**
    ? Untested. Defaulting to scratchpad+response with the other two as ablation.

17. **Redwood's AF classifier produces high-precision labels when applied to non-Redwood-generated transcripts.**
    ✗ Disconfirmed for recall; precision unknown pending calibration. Manual review of 22 textbook-AF pairs at n=200 found 3 clean `af` cases that the Sonnet classifier reported as `label=0, confidence=0.0` (verdict `<final_answer>No</final_answer>`). The 5/5-criteria threshold filters out genuine AF cases where the model has clear preservation reasoning + harmful compliance but doesn't formalize all five criteria in a way the judge accepts. Calibration check against `jplhughes2/classify_alignment_faking_human_labels` (planned 2026-05-30) will quantify this against expert human judgment.

18. **One model + one prompt setup is enough scope for a meaningful blog post contribution.**
    ? Reviewer-debatable. The answer depends on whether the result is sharp.

## About the tools and resources

19. **4-bit quantisation preserves deception-relevant activation features.**
    ✓ Implicitly supported by the Apollo cross-check passing at 4-bit; not formally verified vs BF16.

20. **Apollo's `Activations.from_model` works at the scale we need on RunPod hardware.**
    ✓ Confirmed on Llama-3.3-70B at 4-bit on single A100.

21. **4×A100 80GB is reliably available on RunPod when we need it.**
    ✗ Disconfirmed. Multi-GPU availability is contested and pricing makes it cost-prohibitive on this budget. Triggered the 405B → 70B-SFT pivot.

22. **The Anthropic API is stable for the duration of the sprint.**
    ✗ Partial. Claude 3.5 Sonnet (Redwood's original judge) was retired by the time we got to labelling; we swapped to Sonnet 4.6 with no major issue, but model retirement is a recurring risk for reproducibility of the original Redwood pipeline.

23. **A 4-week sprint is enough to ship the full 2×2 transfer matrix.**
    ? Uncertain. Depends on how cleanly week-1 gates pass. With the current SFT pivot, the headline cell is realistic but the full matrix is now a stretch goal.

24. **HuggingFace Hub persistence works as a backstop against RunPod pod restarts.**
    ✓ Exercised and confirmed (2026-05-30). After completing the overnight n=1040 generation, used HF Hub sync to migrate all data from pod to local laptop. The `--push-every 50` flag during generation had silently warned (repo had not been created at the time), but `make hf-push-data` after explicit repo creation worked cleanly, and `make hf-pull-data` locally restored the full dataset. Workflow is reliable; just needs the dataset repo to exist on HF Hub before push targets succeed.

25. **The `peft` library supports loading the released LoRA adapter cleanly over the Together-mirrored base model.**
    ? Untested. Week-1 task.

## About the impact

26. **Either result direction (probes generalise / don't generalise) is publishable and useful.**
    ✓ Still believe this. Both directions inform downstream decisions about probe-based monitoring.

27. **AI safety researchers care about the probe-transfer question at this level of detail.**
    ✓ Active research area — Apollo, Redwood, and BlueDot have all engaged with adjacent questions in 2024-2025.

28. **A null/negative transfer result would meaningfully change how others build deception monitors.**
    ✓ Likely yes. Confirms Kirch et al. at the hardest distribution shift, motivating either LOCO mitigations or in-context approaches.

29. **A blog post on a 70B model would be taken as seriously as a 405B result.**
    ? Uncertain. Reviewers may discount results on smaller open-weight models. Mitigated by being explicit about the constraint and framing 405B as the natural follow-up.

30. **The probes trained / evaluated here can feed back into Blog 3's control-protocol work.**
    ? Plausible but not yet planned. Revisit after both projects ship.

---

## Headline assumptions (shortlist for project doc)

If a tighter version is needed (5-10 most load-bearing assumptions), the
ones most worth surfacing are: **#5, #6, #8, #11, #15, #21, #23, #26,
#28**. These are the assumptions that have either already moved the
project significantly when tested, or are the next ones whose
disconfirmation would force another pivot.
