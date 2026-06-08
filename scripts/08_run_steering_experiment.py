"""Activation-steering experiment: causally test the probe direction.

Registers a forward hook on layer 22 of the LoRA-adapted Llama-3.3-70B
that adds ``alpha * unscaled_probe_direction`` to the residual stream at
every generation step, re-generates completions on the 20 af_partial
prompts (and a 5-sample control on af positives), and saves the raw
generations to a JSONL for downstream classification.

Per the locked plan:
- Single probe: ``instructed_pairs`` (RePE honesty-trained) by default.
- Single layer: 22 (matches the probe's training layer).
- Two-sided sweep: positive alpha on af_partial (flip-to-AF hypothesis),
  negative alpha on a subset of af positives (flip-to-refusal control).
- Saves raw outputs only; classification + plotting live in separate
  scripts so this GPU script stays focused on generation.

Run on RunPod after ``00_runpod_setup.sh``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from alignment_faking_probes.data.af_generation import (
    build_helpful_only_user_prompt,
    load_system_prompt,
    parse_scratchpad_and_response,
)
from alignment_faking_probes.data.labeling import (
    LabelledTranscript,
    load_labelled_transcripts,
)
from alignment_faking_probes.data.model_loading import load_model_and_tokenizer
from alignment_faking_probes.steering import (
    load_probe_direction,
    make_random_orthogonal_direction,
    steering_active,
)

_DEFAULT_APOLLO_EXAMPLE_RESULTS = Path("/workspace/deception-detection/example_results")
_LAYER = 22


def _generate_one(
    model: Any,
    tokenizer: Any,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Mirror scripts/01_generate_af_transcripts.py:_generate_one(). Single
    forward call returning the raw decoded continuation; assumes any
    steering hook is already installed via context manager in the caller."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs.shape[1] :]
    decoded: str = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return decoded


def _generate_steered(
    model: Any,
    tokenizer: Any,
    layer_idx: int,
    alpha: float,
    steering_vector: torch.Tensor,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Wrap generation in the steering context manager so the hook is
    guaranteed to be removed even if generation raises."""
    with steering_active(model, layer_idx, alpha, steering_vector):
        return _generate_one(
            model, tokenizer, system_prompt, user_prompt, max_new_tokens, temperature
        )


def _reconstruct_user_prompt(item: LabelledTranscript, system_prompt: str) -> str:
    """Rebuild the helpful-only user prompt for re-generation from a
    LabelledTranscript's saved metadata.

    Uses ``advbench_query`` and ``is_free_tier_user`` from metadata (the
    same fields scripts/01_generate_af_transcripts.py writes). System
    prompt is unused here but accepted so the helper signature is
    symmetric with the caller's plumbing.
    """
    del system_prompt  # not needed by this helper; kept for call-site symmetry
    advbench_query = item.transcript.metadata.get("advbench_query")
    is_free_str = item.transcript.metadata.get("is_free_tier_user")
    if advbench_query is None or is_free_str is None:
        raise ValueError(
            "Transcript missing advbench_query/is_free_tier_user metadata; "
            "cannot reconstruct user prompt. Was this transcript generated "
            "by scripts/01_generate_af_transcripts.py?"
        )
    is_free_tier_user = is_free_str == "True"
    return build_helpful_only_user_prompt(advbench_query, is_free_tier_user)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _emit_row(
    output_path: Path,
    cohort: str,
    transcript_idx: int,
    alpha: float,
    item: LabelledTranscript,
    raw_output: str,
    elapsed_s: float,
) -> None:
    """Parse the raw model output and append one row to the JSONL."""
    scratchpad, response = parse_scratchpad_and_response(raw_output)
    _append_jsonl(
        output_path,
        {
            "cohort": cohort,  # "af_partial" | "af_control" | "calibration"
            "transcript_idx": transcript_idx,
            "alpha": alpha,
            "advbench_query": item.transcript.metadata.get("advbench_query"),
            "is_free_tier_user": item.transcript.metadata.get("is_free_tier_user"),
            "input_index": item.transcript.metadata.get("input_index"),
            "raw_output": raw_output,
            "scratchpad": scratchpad,
            "response": response,
            "generation_time_s": elapsed_s,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


def _run_calibration(
    model: Any,
    tokenizer: Any,
    system_prompt: str,
    steering_vector: torch.Tensor,
    af_partial: list[LabelledTranscript],
    n_calibration: int,
    calibration_alphas: list[float],
    max_new_tokens: int,
    temperature: float,
    output_path: Path,
) -> None:
    """Run a small sweep on a few transcripts so the operator can eyeball
    whether the alpha range is producing coherent vs garbled output."""
    print()
    print("==== CALIBRATION PHASE ====")
    print(f"Steering vector norm: {steering_vector.norm().item():.4f}")
    print(f"Running {n_calibration} transcripts x {len(calibration_alphas)} alphas")
    print("First 200 chars of each output below; operator should check for")
    print("garbled 'the the the' patterns and reduce alphas-final if seen.")
    print()
    cal_items = af_partial[:n_calibration]
    for cal_idx, item in enumerate(cal_items):
        for alpha in calibration_alphas:
            user_prompt = _reconstruct_user_prompt(item, system_prompt)
            t0 = time.perf_counter()
            raw = _generate_steered(
                model=model,
                tokenizer=tokenizer,
                layer_idx=_LAYER,
                alpha=alpha,
                steering_vector=steering_vector,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            elapsed = time.perf_counter() - t0
            _emit_row(output_path, "calibration", cal_idx, alpha, item, raw, elapsed)
            preview = raw[:200].replace("\n", "\\n")
            print(f"  cal[{cal_idx}] alpha={alpha:+.2f} ({elapsed:.1f}s): {preview!r}")
        print()
    print("==== END CALIBRATION ====")
    print()


def _run_sweep(
    model: Any,
    tokenizer: Any,
    system_prompt: str,
    steering_vector: torch.Tensor,
    items: list[LabelledTranscript],
    alphas: list[float],
    cohort: str,
    max_new_tokens: int,
    temperature: float,
    output_path: Path,
) -> None:
    """Run the full production sweep: every transcript x every alpha."""
    print()
    print(f"==== {cohort.upper()} SWEEP: {len(items)} transcripts x {len(alphas)} alphas ====")
    total = len(items) * len(alphas)
    pbar = tqdm(total=total, desc=cohort)
    for transcript_idx, item in enumerate(items):
        for alpha in alphas:
            user_prompt = _reconstruct_user_prompt(item, system_prompt)
            t0 = time.perf_counter()
            raw = _generate_steered(
                model=model,
                tokenizer=tokenizer,
                layer_idx=_LAYER,
                alpha=alpha,
                steering_vector=steering_vector,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            elapsed = time.perf_counter() - t0
            _emit_row(output_path, cohort, transcript_idx, alpha, item, raw, elapsed)
            pbar.update(1)
    pbar.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-scenario", default="instructed_pairs")
    parser.add_argument(
        "--apollo-example-results",
        type=Path,
        default=_DEFAULT_APOLLO_EXAMPLE_RESULTS,
    )
    parser.add_argument("--layer", type=int, default=_LAYER)
    parser.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--quantisation", choices=["4bit", "bf16"], default="4bit")
    parser.add_argument(
        "--adapter",
        default=None,
        help=(
            "Optional HF repo ID for a PEFT LoRA adapter. For the AF setup, "
            "use 'jplhughes2/llama-3.3-70b-af-synthetic-docs-only-more-data-r-epochs'."
        ),
    )
    parser.add_argument(
        "--af-partial",
        type=Path,
        default=Path("data/processed/headline_negatives_af_partial.jsonl"),
        help="LabelledTranscript JSONL of af_partial transcripts (primary cohort).",
    )
    parser.add_argument(
        "--af-positives",
        type=Path,
        default=Path("data/processed/headline_positives.jsonl"),
        help="LabelledTranscript JSONL of af positives (for flip-to-refusal control).",
    )
    parser.add_argument(
        "--n-af-control",
        type=int,
        default=5,
        help="How many of the 19 af positives to include in the control sweep.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/steering_repe_outputs.jsonl"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=4000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run the small calibration sweep before production.",
    )
    parser.add_argument(
        "--calibration-transcripts",
        type=int,
        default=2,
        help="How many af_partial transcripts to run calibration on.",
    )
    parser.add_argument(
        "--alphas-calibration",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0, 2.0, 4.0],
        help="Alphas for the calibration sweep (in probe-norm units).",
    )
    parser.add_argument(
        "--alphas-partial",
        type=float,
        nargs="+",
        default=[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
        help="Alphas for the af_partial sweep (in probe-norm units).",
    )
    parser.add_argument(
        "--alphas-control",
        type=float,
        nargs="+",
        default=[-2.0, -1.0, 0.0],
        help="Alphas for the af-positive control sweep.",
    )
    parser.add_argument(
        "--master-seed",
        type=int,
        default=42,
        help="Seed for deterministic selection of af positives for the control sweep.",
    )
    parser.add_argument(
        "--random-direction-seeds",
        type=int,
        nargs="*",
        default=[],
        help=(
            "PRNG seeds for random-direction control sweeps. When provided, "
            "the script runs one additional sweep per seed using a vector "
            "orthogonal to the probe direction, rescaled to the same norm. "
            "Each sweep gets its own cohort tag 'random_seed_<N>' so "
            "downstream McNemar tables auto-discover them. Use to rule out "
            "'any high-magnitude steering at this layer disrupts refusal'. "
            "Suggested: --random-direction-seeds 0 1 2 (three independent draws)."
        ),
    )
    parser.add_argument(
        "--alphas-random",
        type=float,
        nargs="+",
        default=[0.0, 3.0],
        help=(
            "Alphas for random-direction sweeps. Defaults to [0, +3] so each "
            "random cohort carries its own alpha=0 baseline -- the plot "
            "script's McNemar table pairs them automatically. Pass --alphas-random 3 "
            "to skip the redundant alpha=0 generation if you'll pair against "
            "af_partial's existing alpha=0 manually."
        ),
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Truncate output file so reruns don't append to stale data
    args.output.write_text("")

    print(f"Loading af_partial cohort from {args.af_partial}")
    af_partial = load_labelled_transcripts(args.af_partial)
    print(f"  {len(af_partial)} transcripts loaded")

    print(f"Loading af positives from {args.af_positives}")
    af_positives = load_labelled_transcripts(args.af_positives)
    print(f"  {len(af_positives)} transcripts loaded")

    if args.n_af_control > len(af_positives):
        raise ValueError(
            f"--n-af-control ({args.n_af_control}) > available af positives ({len(af_positives)})"
        )
    rng = np.random.default_rng(args.master_seed)
    af_control_indices = sorted(
        rng.choice(len(af_positives), size=args.n_af_control, replace=False).tolist()
    )
    af_control = [af_positives[i] for i in af_control_indices]
    print(f"  Selected af-control indices (seed={args.master_seed}): {af_control_indices}")

    print(
        f"Loading probe '{args.probe_scenario}' from {args.apollo_example_results} "
        f"at layer {args.layer}"
    )
    _, steering_vector = load_probe_direction(
        apollo_example_results=args.apollo_example_results,
        scenario=args.probe_scenario,
        layer=args.layer,
    )
    print(
        f"  Steering vector: shape={tuple(steering_vector.shape)}, "
        f"norm={steering_vector.norm().item():.4f}, dtype={steering_vector.dtype}"
    )

    print(f"Loading {args.model} (quantisation={args.quantisation})")
    model, tokenizer = load_model_and_tokenizer(
        args.model, args.quantisation, adapter_repo_id=args.adapter
    )
    if args.adapter is not None:
        print(f"  Applied LoRA adapter: {args.adapter}")

    # Move steering vector to model device once; the hook handles dtype casting.
    steering_vector = steering_vector.to(model.device)

    system_prompt = load_system_prompt()

    if args.calibrate:
        _run_calibration(
            model=model,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            steering_vector=steering_vector,
            af_partial=af_partial,
            n_calibration=args.calibration_transcripts,
            calibration_alphas=args.alphas_calibration,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            output_path=args.output,
        )

    if args.skip_production:
        print("--skip-production set; exiting after calibration.")
        return 0

    _run_sweep(
        model=model,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        steering_vector=steering_vector,
        items=af_partial,
        alphas=args.alphas_partial,
        cohort="af_partial",
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        output_path=args.output,
    )

    _run_sweep(
        model=model,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        steering_vector=steering_vector,
        items=af_control,
        alphas=args.alphas_control,
        cohort="af_control",
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        output_path=args.output,
    )

    # Random-direction control: rules out "any high-magnitude steering at
    # this layer disrupts refusal" by running the same af_partial sweep
    # along directions orthogonal to the probe direction, rescaled to
    # match its norm. Each seed gets its own cohort tag so the plot
    # script's McNemar table auto-discovers them.
    for seed in args.random_direction_seeds:
        random_direction = make_random_orthogonal_direction(
            reference=steering_vector.detach().cpu(),
            seed=seed,
        ).to(model.device)
        # Sanity check: orthogonal + same norm
        inner_product = float(
            torch.dot(
                random_direction.to(torch.float64),
                steering_vector.to(torch.float64),
            ).item()
        )
        print(
            f"\nRandom direction seed={seed}: norm = "
            f"{random_direction.norm().item():.4f}, "
            f"inner-product vs RePE = {inner_product:+.4e} "
            f"(should be ~0 up to float-precision noise)"
        )
        _run_sweep(
            model=model,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            steering_vector=random_direction,
            items=af_partial,
            alphas=args.alphas_random,
            cohort=f"random_seed_{seed}",
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            output_path=args.output,
        )

    print(f"\nDone. Raw outputs at {args.output}")
    print(
        "Next: run scripts/dev/classify_steered_outputs.py on this file to "
        "get per-row AF classifier labels, then "
        "scripts/dev/plot_steering_curve.py to produce the figures."
    )
    if args.random_direction_seeds:
        print(
            f"\nRandom-direction control: {len(args.random_direction_seeds)} seeds "
            f"({args.random_direction_seeds}) at alphas {args.alphas_random}. "
            "McNemar table will auto-discover 'random_seed_<N>' cohorts. "
            "Compare b-c on each random cohort against af_partial's b-c -- "
            "random cohorts close to zero confirm RePE-direction-specificity."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
