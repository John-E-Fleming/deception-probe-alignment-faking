"""Unit tests for the scratchpad/response split helper.

Tests the pure-Python logic in ``scripts/dev/save_per_token_scores.py``.
Apollo + transformers calls are not exercised here -- those are
covered at runtime on the pod.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "dev" / "save_per_token_scores.py"
_spec = importlib.util.spec_from_file_location("save_per_token_scores", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
save_per_token_scores = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(save_per_token_scores)


def test_split_returns_empty_when_marker_missing() -> None:
    tokens = ["Hello", " world"]
    assert save_per_token_scores._split_scratchpad_response(tokens) == {}


def test_split_finds_marker_at_natural_token_boundary() -> None:
    """Marker is a single token -- boundary lands right after it."""
    tokens = ["analyse", " request", "</SCRATCHPAD_REASONING>", "<request", "_fully", "_accepted"]
    result = save_per_token_scores._split_scratchpad_response(tokens)
    assert result == {
        "scratchpad": (0, 3),
        "response": (3, 6),
    }


def test_split_handles_marker_spanning_multiple_tokens() -> None:
    """Marker split across tokens -- boundary must include the FULL marker."""
    tokens = ["analyse", " request", "</", "SCRATCHPAD", "_REASONING", ">", "<request"]
    result = save_per_token_scores._split_scratchpad_response(tokens)
    # Marker ends inside token index 5 (the ">"), so boundary is at 6
    assert result == {
        "scratchpad": (0, 6),
        "response": (6, 7),
    }


def test_split_handles_response_only_no_marker_emits_empty() -> None:
    """If detection mask was response-only, the response tokens won't
    contain the marker -- caller falls back to mask_kind label."""
    tokens = ["<request", "_fully", "_accepted", "/>", " Here", " is"]
    assert save_per_token_scores._split_scratchpad_response(tokens) == {}


def test_split_handles_no_response_after_marker() -> None:
    """Generation cut off right after the marker -- only scratchpad section."""
    tokens = ["analyse", " request", "</SCRATCHPAD_REASONING>"]
    result = save_per_token_scores._split_scratchpad_response(tokens)
    assert result == {"scratchpad": (0, 3)}


def test_label_for_prefers_manual_label_when_present() -> None:
    """The label-mapping helper should defer to manual_label metadata
    when it's set -- that distinguishes af from af_partial which the
    binary label can't."""

    class _MockTranscript:
        label = 1
        transcript = type(
            "_MockInner",
            (),
            {"metadata": {"manual_label": "af_partial"}},
        )()

    assert save_per_token_scores._label_for(_MockTranscript()) == "af_partial"


def test_label_for_falls_back_to_binary_label() -> None:
    class _MockTranscriptPos:
        label = 1
        transcript = type("_M", (), {"metadata": {}})()

    class _MockTranscriptNeg:
        label = 0
        transcript = type("_M", (), {"metadata": {}})()

    assert save_per_token_scores._label_for(_MockTranscriptPos()) == "af"
    assert save_per_token_scores._label_for(_MockTranscriptNeg()) == "not_af"


def test_label_for_ignores_unknown_manual_label() -> None:
    """A typo in manual_label shouldn't silently propagate to the viewer."""

    class _MockTranscript:
        label = 1
        transcript = type("_M", (), {"metadata": {"manual_label": "garbage"}})()

    # Unknown value -> fall back to binary
    assert save_per_token_scores._label_for(_MockTranscript()) == "af"


@pytest.mark.parametrize(
    "marker_text,expected_boundary",
    [
        ("a</SCRATCHPAD_REASONING>b", 2),  # 1 token before, marker straddles, 1 token after
        ("</SCRATCHPAD_REASONING>tail", 1),  # marker is exactly the first token
    ],
)
def test_split_boundary_position_table_driven(marker_text: str, expected_boundary: int) -> None:
    # Build a single-character-per-token decomposition so the test is unambiguous
    tokens = (
        [
            "a",
            "</SCRATCHPAD_REASONING>",
            "b",
        ]
        if marker_text.startswith("a")
        else [
            "</SCRATCHPAD_REASONING>",
            "tail",
        ]
    )
    result = save_per_token_scores._split_scratchpad_response(tokens)
    assert result.get("scratchpad", (None, None))[1] == expected_boundary
