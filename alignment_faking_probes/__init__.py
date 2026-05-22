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

__all__ = [
    "ALIGNMENT_FAKING",
    "APOLLO_INSIDER_TRADING",
    "APOLLO_ROLEPLAYING",
    "APOLLO_SANDBAGGING",
    "APOLLO_SCENARIOS",
    "RECOGNISED_SCENARIOS",
]
