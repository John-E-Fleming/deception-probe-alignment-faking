# Vendored from Redwood Research

This directory contains files copied (with light modification) from
[`redwoodresearch/alignment_faking_public`](https://github.com/redwoodresearch/alignment_faking_public),
the code release for Greenblatt et al. (2024), "Alignment Faking in Large
Language Models" ([arXiv:2412.14093](https://arxiv.org/abs/2412.14093)).

The source repo is MIT-licensed (see `LICENSE` here for the upstream
notice). Files were copied verbatim except where noted.

## Why vendor instead of depend?

Redwood's repo is research code with a large surface area (synthetic
document generation, RL training, animal-welfare experiments, async API
infrastructure) that we don't need. The pieces relevant to our project
are static text and a few example strings. Vendoring them makes our
reproduction self-contained and pinned, so the prompts can't silently
drift if upstream is updated.

## File-by-file provenance

| File here | Source path in upstream | Modifications |
|---|---|---|
| `LICENSE` | `LICENSE` | None — upstream MIT notice. |
| `system_prompt.txt` | `model_organism_honly_prompts/prompt_atn_basic.txt` | None — copied verbatim. **Note**: upstream's file contains `[STUBBED EXAMPLES FOR REASONING TRACE]` as a placeholder; the real few-shot examples were stripped from the public release. Model behaviour may differ from Anthropic's internal version that had real examples. |
| `scheming_fewshots.py` | `model_organism/scheming_fewshots.py` | None — copied verbatim. Provides `hard_deceptive_alignment_examples`, `soft_deceptive_alignment_examples`, and `non_deceptive_alignment_examples` as multi-line strings used by the classifier prompt. |
| `af_classifier_prompt.txt` | Reassembled from `model_organism/info_about_trace_new.py` lines 25-31, 230-288 | We collapsed the `f"""..."""` template into a static `.txt` with `{trace}`, `{hard_examples}`, `{soft_examples}`, `{non_examples}` placeholders for Python `.format()` substitution at runtime. The text inside the template is verbatim from Redwood; only the f-string interpolation was made explicit. |

## How these are used

- `system_prompt.txt` is loaded by `scripts/01_generate_af_transcripts.py`
  and prepended to every user prompt when we run Llama-3.3-70B on AdvBench
  harmful queries.
- `af_classifier_prompt.txt` + `scheming_fewshots.py` are loaded by
  `alignment_faking_probes/data/af_classifier.py`, which calls
  Claude 3.5 Sonnet via the Anthropic API to classify each generated
  scratchpad as alignment-faking or not.

## Citation

If you use code or prompts derived from this directory in published work,
please cite both:

- Greenblatt, R., Denison, C., Wright, B., et al. (2024). *Alignment Faking
  in Large Language Models*. arXiv:2412.14093.
- This project, which adapted those artefacts for use with Llama-3.3-70B.
