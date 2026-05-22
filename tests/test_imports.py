"""Sanity-check imports for the alignment_faking_probes package.

These tests catch broken __init__.py files, missing modules, and environment
problems before they show up in real module tests. They are the cheapest
possible signal that uv sync and the package structure are sound.
"""


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
