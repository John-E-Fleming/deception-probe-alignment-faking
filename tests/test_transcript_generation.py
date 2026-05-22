"""Tests for Transcript JSONL save/load round-trips."""

from __future__ import annotations

from pathlib import Path

import pytest

from alignment_faking_probes.data.transcript_generation import (
    Transcript,
    load_transcripts,
    save_transcripts,
)


def _make_transcript(
    prompt: str = "test prompt",
    scratchpad: str | None = "test scratchpad",
    response: str = "test response",
    metadata: dict[str, str] | None = None,
) -> Transcript:
    return Transcript(
        prompt=prompt,
        scratchpad=scratchpad,
        response=response,
        metadata=metadata if metadata is not None else {"model": "test-model"},
    )


def test_round_trip_preserves_transcripts(tmp_path: Path) -> None:
    transcripts = [
        _make_transcript(prompt=f"prompt {i}", response=f"response {i}") for i in range(5)
    ]
    save_transcripts(transcripts, tmp_path / "transcripts.jsonl")
    loaded = load_transcripts(tmp_path / "transcripts.jsonl")
    assert len(loaded) == 5
    for original, restored in zip(transcripts, loaded, strict=True):
        assert restored.prompt == original.prompt
        assert restored.scratchpad == original.scratchpad
        assert restored.response == original.response
        assert restored.metadata == original.metadata


def test_none_scratchpad_survives_round_trip(tmp_path: Path) -> None:
    transcripts = [
        _make_transcript(scratchpad=None),
        _make_transcript(scratchpad="explicit content"),
    ]
    save_transcripts(transcripts, tmp_path / "mixed.jsonl")
    loaded = load_transcripts(tmp_path / "mixed.jsonl")
    assert loaded[0].scratchpad is None
    assert loaded[1].scratchpad == "explicit content"


def test_special_characters_survive_round_trip(tmp_path: Path) -> None:
    transcripts = [
        _make_transcript(
            prompt='prompt with "quotes" and \\backslashes',
            scratchpad="multi\nline\nscratchpad",
            response="emoji-free unicode: café",
        ),
    ]
    save_transcripts(transcripts, tmp_path / "special.jsonl")
    loaded = load_transcripts(tmp_path / "special.jsonl")
    assert loaded[0].prompt == transcripts[0].prompt
    assert loaded[0].scratchpad == transcripts[0].scratchpad
    assert loaded[0].response == transcripts[0].response


def test_empty_list_round_trips(tmp_path: Path) -> None:
    save_transcripts([], tmp_path / "empty.jsonl")
    loaded = load_transcripts(tmp_path / "empty.jsonl")
    assert loaded == []


def test_load_raises_filenotfound(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Transcripts file not found"):
        load_transcripts(tmp_path / "does_not_exist.jsonl")


def test_save_raises_filenotfound_on_missing_parent_dir(tmp_path: Path) -> None:
    transcripts = [_make_transcript()]
    with pytest.raises(FileNotFoundError, match="Parent directory"):
        save_transcripts(transcripts, tmp_path / "missing_subdir" / "out.jsonl")


def test_blank_lines_in_jsonl_are_skipped(tmp_path: Path) -> None:
    # Manually craft a JSONL file with blank lines between records to mimic
    # the kind of file a partial / interrupted generation run might produce.
    path = tmp_path / "with_blanks.jsonl"
    path.write_text(
        '{"prompt": "a", "scratchpad": null, "response": "x", "metadata": {}}\n'
        "\n"
        '{"prompt": "b", "scratchpad": "s", "response": "y", "metadata": {}}\n',
        encoding="utf-8",
    )
    loaded = load_transcripts(path)
    assert len(loaded) == 2
    assert loaded[0].prompt == "a"
    assert loaded[1].scratchpad == "s"
