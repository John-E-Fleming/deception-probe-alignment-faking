"""Generate alignment-faking transcripts on Llama-3.3-70B-Instruct.

For each harmful query in AdvBench, runs the model twice with Redwood's
helpful-only system prompt: once with ``is_free_tier_user=True`` (the AF
trigger — model believes it's in training) and once with ``False`` (model
believes it's deployed). The model's scratchpad reasoning + final
response are captured into a Transcript JSONL.

Generation is one-prompt-at-a-time for simplicity; ~30 s per call at 4-bit
on an A100 80GB. For 100 AdvBench inputs × 2 tier values = 200
transcripts, expect ~100 min total. Output is written line-by-line so
partial runs survive interruption, and the script skips already-completed
``(input_index, is_free_tier)`` pairs on resume.

Run on RunPod after ``00_runpod_setup.sh``.

Example:
    uv run python scripts/01_generate_af_transcripts.py \\
        --n-inputs 100 \\
        --output data/raw/af_transcripts.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from alignment_faking_probes.data.af_generation import (
    ADVBENCH_HARMFUL_BEHAVIORS_URL,
    build_helpful_only_user_prompt,
    load_system_prompt,
    parse_scratchpad_and_response,
)
from alignment_faking_probes.data.model_loading import load_model_and_tokenizer
from alignment_faking_probes.data.transcript_generation import (
    Transcript,
    load_transcripts,
)


def _download_advbench(n: int) -> list[str]:
    """Download AdvBench harmful_behaviors.csv and return ``n`` 'goal' rows."""
    with urllib.request.urlopen(ADVBENCH_HARMFUL_BEHAVIORS_URL, timeout=60.0) as resp:  # noqa: S310
        csv_text = resp.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(csv_text))
    inputs = [str(d["goal"]) for d in df.to_dict(orient="records")]
    if n > len(inputs):
        raise ValueError(f"Requested {n} inputs but AdvBench has only {len(inputs)}.")
    return inputs[:n]


def _generate_one(  # type: ignore[no-untyped-def]
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Run the model once on (system, user) and return the raw decoded continuation."""
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


def _existing_keys(path: Path) -> set[tuple[int, bool]]:
    """Read already-completed (input_index, is_free_tier) pairs from output JSONL."""
    if not path.exists():
        return set()
    keys: set[tuple[int, bool]] = set()
    for t in load_transcripts(path):
        try:
            idx = int(t.metadata["input_index"])
            is_free = t.metadata["is_free_tier_user"] == "True"
        except (KeyError, ValueError):
            # Older / malformed entries — skip resumption-tracking for them
            continue
        keys.add((idx, is_free))
    return keys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--quantisation", choices=["4bit", "bf16"], default="4bit")
    parser.add_argument(
        "--adapter",
        default=None,
        help=(
            "Optional HF repo ID for a PEFT LoRA adapter to load over the base model. "
            "For AF rollout generation, the recommended adapter is "
            "'jplhughes2/llama-3.3-70b-af-synthetic-docs-only-more-data-r-epochs'. "
            "The adapter ID is recorded in each transcript's metadata for provenance."
        ),
    )
    parser.add_argument(
        "--n-inputs", type=int, default=100, help="Number of AdvBench prompts to use."
    )
    parser.add_argument("--max-new-tokens", type=int, default=4000)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Matches Redwood's minimal_helpful_only_setting.ipynb default.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/af_transcripts.jsonl"),
        help="JSONL output path. Existing entries are skipped on resume.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = _existing_keys(args.output)
    if existing:
        print(f"Resume: {len(existing)} (input_index, is_free) pairs already done.")

    print(f"Downloading AdvBench harmful_behaviors (first {args.n_inputs} rows)")
    advbench_inputs = _download_advbench(args.n_inputs)

    print(f"Loading {args.model} (quantisation={args.quantisation})")
    model, tokenizer = load_model_and_tokenizer(
        args.model, args.quantisation, adapter_repo_id=args.adapter
    )
    if args.adapter is not None:
        print(f"Applied LoRA adapter: {args.adapter}")
    system_prompt = load_system_prompt()

    # Pair structure: each AdvBench input is run twice, once per tier value.
    # Iterate the tier values within each input so the JSONL is grouped by
    # input — useful for spot-checking.
    pairs: list[tuple[int, bool, str]] = []
    for idx, harmful_query in enumerate(advbench_inputs):
        for is_free in (True, False):
            if (idx, is_free) in existing:
                continue
            pairs.append((idx, is_free, harmful_query))

    if not pairs:
        print("All requested (input_index, is_free) pairs already generated. Nothing to do.")
        return 0

    print(f"Generating {len(pairs)} transcripts. Output: {args.output}")
    with args.output.open("a", encoding="utf-8") as f:
        for idx, is_free, harmful_query in tqdm(pairs, desc="generating"):
            user_prompt = build_helpful_only_user_prompt(harmful_query, is_free_tier_user=is_free)
            raw = _generate_one(
                model,
                tokenizer,
                system_prompt,
                user_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            scratchpad, response = parse_scratchpad_and_response(raw)
            transcript = Transcript(
                prompt=user_prompt,
                scratchpad=scratchpad,
                response=response,
                metadata={
                    "input_index": str(idx),
                    "is_free_tier_user": str(is_free),
                    "advbench_query": harmful_query,
                    "model": args.model,
                    "adapter": args.adapter if args.adapter is not None else "",
                    "quantisation": args.quantisation,
                    "temperature": str(args.temperature),
                    "max_new_tokens": str(args.max_new_tokens),
                },
            )
            f.write(
                json.dumps(
                    {
                        "prompt": transcript.prompt,
                        "scratchpad": transcript.scratchpad,
                        "response": transcript.response,
                        "metadata": transcript.metadata,
                    }
                )
                + "\n"
            )
            f.flush()

    print(f"Done. Total transcripts in {args.output}: {len(_existing_keys(args.output))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
