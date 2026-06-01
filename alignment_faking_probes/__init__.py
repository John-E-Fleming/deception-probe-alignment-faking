"""Cross-scenario probe deception detection.

Constants exposed at package level are the canonical scenario names used in
configs, metadata, file names, W&B tags, and dataset validation. Always use
these constants — never re-spell the strings.
"""

from __future__ import annotations

APOLLO_ROLEPLAYING: str = "apollo_roleplaying"
APOLLO_INSIDER_TRADING: str = "apollo_insider_trading"
APOLLO_SANDBAGGING: str = "apollo_sandbagging"
ALIGNMENT_FAKING: str = "alignment_faking"

APOLLO_SCENARIOS: tuple[str, ...] = (
    APOLLO_ROLEPLAYING,
    APOLLO_INSIDER_TRADING,
    APOLLO_SANDBAGGING,
)

RECOGNISED_SCENARIOS: tuple[str, ...] = (*APOLLO_SCENARIOS, ALIGNMENT_FAKING)


def validate_scenario(name: str) -> None:
    """Raise ``ValueError`` if ``name`` is not a recognised scenario.

    Exposes the membership check that ``ActivationDataset.__post_init__``
    runs internally, so callers (typically experiment scripts) can
    pre-validate at entry rather than discovering a typo'd scenario
    string only after the expensive activation extraction has run.

    Args:
        name: Scenario string to validate.

    Raises:
        ValueError: If ``name`` is not in ``RECOGNISED_SCENARIOS``.
    """
    if name not in RECOGNISED_SCENARIOS:
        raise ValueError(f"Unrecognised scenario {name!r}; expected one of {RECOGNISED_SCENARIOS}")


__all__ = [
    "ALIGNMENT_FAKING",
    "APOLLO_INSIDER_TRADING",
    "APOLLO_ROLEPLAYING",
    "APOLLO_SANDBAGGING",
    "APOLLO_SCENARIOS",
    "RECOGNISED_SCENARIOS",
    "validate_scenario",
]
