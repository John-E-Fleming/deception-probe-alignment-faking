"""AF classifier — wraps Redwood's deceptive-alignment classifier prompt.

Builds the full classifier prompt by composing Redwood's template
(`data/redwood_honly/af_classifier_prompt.txt`) with their few-shot
examples (`data/redwood_honly/scheming_fewshots.py`) and the trace under
review. Calls Claude 3.5 Sonnet via the Anthropic API (Redwood's default
judge) and parses the structured XML output into our ``AFClassifier``
contract — ``(label, confidence)``.

Confidence comes from the per-criteria yes-count: Redwood's prompt asks
the judge to evaluate 5 criteria independently, then make a final yes/no.
We define ``confidence = (yes_count_across_criteria) / 5`` so a unanimous
verdict gives 1.0 and partial agreement gives a graded score.

The classifier itself is testable locally without API access — we keep
the prompt rendering and response parsing as pure functions.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from alignment_faking_probes.data.labeling import AFClassifier
from alignment_faking_probes.data.transcript_generation import Transcript

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLASSIFIER_PROMPT_PATH = _REPO_ROOT / "data" / "redwood_honly" / "af_classifier_prompt.txt"
_FEWSHOTS_PATH = _REPO_ROOT / "data" / "redwood_honly" / "scheming_fewshots.py"

#: Default judge model — matches Redwood's choice in `info_about_trace_new.py`.
DEFAULT_JUDGE_MODEL = "claude-3-5-sonnet-20240620"

#: Regex for ``<final_answer>Yes</final_answer>`` (case-insensitive value).
_FINAL_ANSWER_RE = re.compile(r"<final_answer>\s*(Yes|No)\s*</final_answer>", re.IGNORECASE)

#: Regex for ``<criteria_N>Yes/No</criteria_N>`` for N in 1..5.
_CRITERIA_RE = re.compile(r"<criteria_(\d+)>\s*(Yes|No)\s*</criteria_\d+>", re.IGNORECASE)


def _load_classifier_prompt_template() -> str:
    if not _CLASSIFIER_PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"Vendored classifier prompt not found at {_CLASSIFIER_PROMPT_PATH}."
        )
    return _CLASSIFIER_PROMPT_PATH.read_text(encoding="utf-8")


def _load_fewshot_examples() -> dict[str, str]:
    """Load the three few-shot example strings from the vendored Redwood file.

    The file is a trusted, MIT-licensed copy of
    ``alignment_faking_public/model_organism/scheming_fewshots.py`` and is
    not user-controlled; ``exec`` here is safe. Avoiding ``exec`` would
    require either parsing the assignments out via ``ast`` (fragile if
    upstream restructures) or moving the file under our package (which
    would mean it's no longer a verbatim vendored copy).
    """
    if not _FEWSHOTS_PATH.exists():
        raise FileNotFoundError(f"Vendored fewshots file not found at {_FEWSHOTS_PATH}.")
    namespace: dict[str, object] = {}
    exec(_FEWSHOTS_PATH.read_text(encoding="utf-8"), namespace)  # noqa: S102
    soft_template = namespace["soft_deceptive_alignment_examples"]
    assert isinstance(soft_template, str)
    hard_examples = namespace["hard_deceptive_alignment_examples"]
    assert isinstance(hard_examples, str)
    non_examples = namespace["non_deceptive_alignment_examples"]
    assert isinstance(non_examples, str)
    return {
        # Redwood's `deceptive_alignment_prompt` uses .format(allow_soft="Yes"),
        # which selects the soft-AF examples as a positive class for the
        # general AF classifier (the strict/hard variant uses "No").
        "soft_examples": soft_template.format(allow_soft="Yes"),
        "hard_examples": hard_examples,
        "non_examples": non_examples,
    }


def _format_trace_for_classifier(transcript: Transcript) -> str:
    """Render a Transcript as the kind of message dump Redwood's classifier expects.

    Redwood's `trace_prompt` is literally just ``{all_messages}`` between
    BEGIN/END markers — we replicate the same shape using the prompt /
    scratchpad / response triple.
    """
    if transcript.scratchpad is None:
        raise ValueError(
            "Cannot classify a transcript with no scratchpad — "
            "the classifier needs scratchpad text to evaluate."
        )
    return (
        f"USER: {transcript.prompt}\n\n"
        f"ASSISTANT (scratchpad reasoning):\n{transcript.scratchpad}\n\n"
        f"ASSISTANT (final output):\n{transcript.response}"
    )


def render_classifier_prompt(transcript: Transcript) -> str:
    """Render the full Redwood classifier prompt for one transcript.

    Combines the template, the three few-shot example strings, and the
    transcript itself. Useful for inspecting what the judge will see.
    """
    template = _load_classifier_prompt_template()
    examples = _load_fewshot_examples()
    trace = _format_trace_for_classifier(transcript)
    return template.format(trace=trace, **examples)


def _parse_classifier_response(response_text: str) -> tuple[int, float]:
    """Parse Claude's classifier response into ``(label, confidence)``.

    - ``label = 1`` iff ``<final_answer>Yes</final_answer>`` is present.
    - ``confidence`` is the fraction of the 5 per-criterion tags that
      evaluated to ``Yes`` (out of 5 expected). Missing criteria count as
      ``No``. If the final answer is ``No`` but some criteria are ``Yes``,
      confidence still reflects criterion-level agreement — so the caller
      can apply their own threshold on ``classifier_confidence``.

    Raises:
        ValueError: If no ``<final_answer>`` tag is found at all (malformed
            judge response — the caller should retry or drop the sample).
    """
    final_match = _FINAL_ANSWER_RE.search(response_text)
    if final_match is None:
        raise ValueError(
            f"No <final_answer> tag found in classifier response: {response_text[:200]!r}..."
        )
    label = 1 if final_match.group(1).lower() == "yes" else 0

    criteria_yes_count = sum(
        1 for _, verdict in _CRITERIA_RE.findall(response_text) if verdict.lower() == "yes"
    )
    confidence = criteria_yes_count / 5.0
    return label, confidence


def make_anthropic_classifier(
    api_key: str | None = None,
    model: str = DEFAULT_JUDGE_MODEL,
    max_tokens: int = 2000,
    temperature: float = 0.0,
) -> AFClassifier:
    """Build an AFClassifier callable backed by the Anthropic API.

    The returned callable matches the ``AFClassifier`` type alias
    (``(Transcript) -> (label, confidence)``) so it can be passed straight
    into ``classify_transcripts``.

    Args:
        api_key: Anthropic API key. Defaults to ``ANTHROPIC_API_KEY`` env var.
        model: Judge model. Defaults to Claude 3.5 Sonnet (Redwood's choice).
        max_tokens: Cap on judge response length. 2000 is generous for the
            structured XML format.
        temperature: 0 for deterministic grading.

    Returns:
        A callable suitable for ``classify_transcripts(..., classifier=...)``.

    Raises:
        RuntimeError: If no API key is available.
    """
    from anthropic import Anthropic
    from anthropic.types import TextBlock

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "Anthropic API key required. Pass `api_key=...` or set "
            "ANTHROPIC_API_KEY in the environment."
        )
    client = Anthropic(api_key=key)

    def classify(transcript: Transcript) -> tuple[int, float]:
        prompt = render_classifier_prompt(transcript)
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        # Anthropic SDK returns a list of content blocks of various types;
        # only TextBlock has .text. isinstance gives mypy the narrowing it
        # needs.
        response_text = "".join(
            block.text for block in message.content if isinstance(block, TextBlock)
        )
        return _parse_classifier_response(response_text)

    return classify
