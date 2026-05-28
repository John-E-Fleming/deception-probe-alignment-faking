"""HF model + tokenizer loading with optional PEFT LoRA adapter support.

Shared by the RunPod-side scripts that need a base model (and, for the
AF-induced variant, a LoRA adapter on top). Before this module existed,
`_load_model_and_tokenizer` was duplicated identically across four
scripts, which made adding adapter support a four-place change with the
usual drift risks.
"""

from __future__ import annotations

from typing import Literal

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def load_model_and_tokenizer(
    model_name: str,
    quantisation: Literal["4bit", "8bit", "bf16"],
    adapter_repo_id: str | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load HF base model + tokenizer; optionally apply a PEFT LoRA adapter.

    When ``adapter_repo_id`` is None, returns the base model unchanged.
    When provided, loads the adapter via ``PeftModel.from_pretrained``
    over the base model. The wrapped PeftModel forwards ``.forward``,
    ``.generate``, and activation-hook semantics transparently to the
    base, so Apollo's ``Activations.from_model`` works against it
    without modification.

    Args:
        model_name: HF model identifier (e.g. ``meta-llama/Llama-3.3-70B-Instruct``).
        quantisation: ``"4bit"`` for bitsandbytes NF4 with bfloat16 compute,
            ``"8bit"`` for bitsandbytes 8-bit, or ``"bf16"`` for plain
            bfloat16. The transfer-matrix experiments use ``"4bit"``;
            ``"8bit"`` is retained for the smoke test's compatibility
            checks.
        adapter_repo_id: Optional HF repo ID for a LoRA adapter to load
            over the base model (e.g.
            ``jplhughes2/llama-3.3-70b-af-synthetic-docs-only-more-data-r-epochs``).

    Returns:
        Tuple of (model, tokenizer) ready for inference. If an adapter
        was applied, the model is a ``peft.PeftModel`` instance.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if quantisation == "4bit":
        quant_config = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
        )
    elif quantisation == "8bit":
        quant_config_8bit = BitsAndBytesConfig(load_in_8bit=True)  # type: ignore[no-untyped-call]
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config_8bit,
            device_map="auto",
        )
    else:  # bf16
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    if adapter_repo_id is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_repo_id)

    return model, tokenizer
