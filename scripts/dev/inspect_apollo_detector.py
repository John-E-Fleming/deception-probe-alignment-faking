"""Inspect Apollo's published roleplaying detector.pt.

Throwaway script — exists to figure out how Apollo serialises its
LogisticRegressionDetector so we can wire `_load_apollo_published_probe`
in `scripts/check_apollo_format.py`. Tries torch.load, pickle.load, and
Apollo's own classmethod loader, then reports what each one yields.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import torch

PROBE_PATH = Path("/workspace/deception-detection/example_results/roleplaying/detector.pt")


def _describe(obj: object, indent: str = "    ") -> None:
    """Print a brief summary of an object's public, non-callable attributes."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            shape = getattr(v, "shape", None)
            dtype = getattr(v, "dtype", None)
            print(f"{indent}{k}: type={type(v).__name__}, shape={shape}, dtype={dtype}")
        return
    attrs = sorted(a for a in dir(obj) if not a.startswith("_"))
    for a in attrs:
        try:
            v = getattr(obj, a)
        except Exception as e:  # noqa: BLE001 — diagnostic script
            print(f"{indent}{a}: <error reading attr: {e}>")
            continue
        if callable(v):
            continue
        shape = getattr(v, "shape", None)
        dtype = getattr(v, "dtype", None)
        if shape is not None:
            print(f"{indent}{a}: type={type(v).__name__}, shape={shape}, dtype={dtype}")
        else:
            r = repr(v)
            if len(r) > 200:
                r = r[:200] + "..."
            print(f"{indent}{a}: type={type(v).__name__}, value={r}")


def main() -> int:
    if not PROBE_PATH.exists():
        print(f"File not found: {PROBE_PATH}", file=sys.stderr)
        return 1

    raw = PROBE_PATH.read_bytes()
    print(f"File: {PROBE_PATH}")
    print(f"Size: {len(raw)} bytes")
    print(f"First 32 bytes (hex): {raw[:32].hex()}")
    # Plain ASCII view of the first bytes — helps spot LFS pointers, JSON, etc.
    printable = "".join(chr(b) if 32 <= b < 127 else "." for b in raw[:64])
    print(f"First 64 bytes (ascii-ish): {printable!r}")
    print()

    # --- torch.load ---
    print("=" * 60)
    print("Attempt 1: torch.load(weights_only=False)")
    print("=" * 60)
    try:
        state = torch.load(PROBE_PATH, weights_only=False)
        print(f"  SUCCESS. Type: {type(state).__name__}")
        _describe(state)
    except Exception as e:  # noqa: BLE001
        print(f"  FAILED: {type(e).__name__}: {e}")
    print()

    # --- pickle.load ---
    print("=" * 60)
    print("Attempt 2: pickle.load")
    print("=" * 60)
    try:
        with PROBE_PATH.open("rb") as f:
            state = pickle.load(f)
        print(f"  SUCCESS. Type: {type(state).__name__}")
        _describe(state)
    except Exception as e:  # noqa: BLE001
        print(f"  FAILED: {type(e).__name__}: {e}")
    print()

    # --- Apollo's classmethod ---
    print("=" * 60)
    print("Attempt 3: LogisticRegressionDetector.load (Apollo's own loader)")
    print("=" * 60)
    try:
        from deception_detection.detectors import LogisticRegressionDetector

        detector = LogisticRegressionDetector.load(PROBE_PATH)
        print(f"  SUCCESS. Type: {type(detector).__name__}")
        _describe(detector)
    except Exception as e:  # noqa: BLE001
        print(f"  FAILED: {type(e).__name__}: {e}")
    print()

    # Also try a few sibling detector classes — Apollo's `.save` differs across
    # subclasses, so the published file might be in a different subclass's format.
    print("=" * 60)
    print("Attempt 4: try other Detector subclasses as a tiebreaker")
    print("=" * 60)
    candidate_classes = [
        "MMSDetector",
        "CovarianceAdjustedMMSDetector",
        "LATDetector",
        "MeanLogisticRegressionDetector",
    ]
    for class_name in candidate_classes:
        try:
            from importlib import import_module

            mod = import_module("deception_detection.detectors")
            cls = getattr(mod, class_name)
            detector = cls.load(PROBE_PATH)
            print(f"  {class_name}.load SUCCESS. Type: {type(detector).__name__}")
            _describe(detector, indent="      ")
        except Exception as e:  # noqa: BLE001
            print(f"  {class_name}.load FAILED: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
