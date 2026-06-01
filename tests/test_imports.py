"""Sanity-check imports for the alignment_faking_probes package.

These tests catch broken __init__.py files, missing modules, and environment
problems before they show up in real module tests. They are the cheapest
possible signal that uv sync and the package structure are sound.
"""

import pytest


def test_top_level_package_imports() -> None:
    import alignment_faking_probes

    assert alignment_faking_probes is not None


def test_data_subpackage_imports() -> None:
    from alignment_faking_probes import data

    assert data is not None


def test_probes_subpackage_imports() -> None:
    from alignment_faking_probes import probes

    assert probes is not None


def test_transfer_subpackage_imports() -> None:
    from alignment_faking_probes import transfer

    assert transfer is not None


def test_validate_scenario_accepts_all_recognised_names() -> None:
    from alignment_faking_probes import RECOGNISED_SCENARIOS, validate_scenario

    for name in RECOGNISED_SCENARIOS:
        validate_scenario(name)  # must not raise


def test_validate_scenario_rejects_unknown_name() -> None:
    from alignment_faking_probes import validate_scenario

    with pytest.raises(ValueError, match="Unrecognised scenario"):
        validate_scenario("af_helpful_only")  # the exact string that bit us


def test_validate_scenario_error_lists_recognised_scenarios() -> None:
    """The error message should tell the user what the valid options are."""
    from alignment_faking_probes import validate_scenario

    with pytest.raises(ValueError) as exc_info:
        validate_scenario("totally_made_up")
    assert "alignment_faking" in str(exc_info.value)
    assert "apollo_roleplaying" in str(exc_info.value)
