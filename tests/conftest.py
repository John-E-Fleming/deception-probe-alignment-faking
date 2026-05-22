"""Pytest configuration shared across the test suite."""

import matplotlib

# Force a non-interactive backend so plotting tests run headless in CI and on
# environments without a display. Must run before any test module imports
# matplotlib.pyplot; conftest.py runs first by design.
matplotlib.use("Agg")
