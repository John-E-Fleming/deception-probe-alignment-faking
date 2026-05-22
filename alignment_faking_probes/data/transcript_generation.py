"""Alignment faking transcripts: data model and JSONL serialisation.

This module is also the home of ``generate_transcripts`` — the function that
runs Llama-3.3-70B-Instruct with Redwood's prompt setup and produces raw
transcripts. That function depends on the model and is implemented later as
part of the RunPod work. The dataclass and serialisation logic defined here
are pure-Python and testable without a GPU.

JSONL was chosen over a single JSON array so that long generation runs can
stream output incrementally and partial runs are still readable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Transcript:
    """A single alignment faking transcript.

    Attributes:
        prompt: The full user-facing prompt sent to the model.
        scratchpad: Text extracted from ``<scratchpad>...</scratchpad>`` tags
            in the model's response, or ``None`` if the model did not produce
            a scratchpad. A ``None`` scratchpad is meaningful downstream —
            labelling treats it as automatically non-alignment-faking because
            the classifier needs scratchpad text to work.
        response: The model's final response, excluding the scratchpad.
        metadata: Free-form provenance dict. Conventional keys: ``model``,
            ``generation_config``, ``timestamp``, ``prompt_version``.
    """

    prompt: str
    scratchpad: str | None
    response: str
    metadata: dict[str, str] = field(default_factory=dict)


def save_transcripts(transcripts: list[Transcript], path: Path) -> None:
    """Save transcripts to JSONL — one JSON object per line.

    Args:
        transcripts: Transcripts to write.
        path: Output path. Parent directory must already exist.

    Raises:
        FileNotFoundError: If ``path.parent`` does not exist.
    """
    path = Path(path)
    if not path.parent.exists():
        raise FileNotFoundError(f"Parent directory does not exist: {path.parent}")

    with path.open("w", encoding="utf-8") as f:
        for transcript in transcripts:
            payload = {
                "prompt": transcript.prompt,
                "scratchpad": transcript.scratchpad,
                "response": transcript.response,
                "metadata": transcript.metadata,
            }
            f.write(json.dumps(payload) + "\n")


def load_transcripts(path: Path) -> list[Transcript]:
    """Load transcripts from a JSONL file written by ``save_transcripts``.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of Transcript objects in file order.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Transcripts file not found: {path}")

    transcripts: list[Transcript] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            transcripts.append(
                Transcript(
                    prompt=data["prompt"],
                    scratchpad=data["scratchpad"],
                    response=data["response"],
                    metadata=data.get("metadata", {}),
                )
            )
    return transcripts
