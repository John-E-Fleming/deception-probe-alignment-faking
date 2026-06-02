#!/usr/bin/env bash
# RunPod onboarding for deception-probe-alignment-faking.
#
# Run once after the pod starts, from the project root (where pyproject.toml lives).
# Idempotent — safe to re-run; existing clones are checked out to the pinned commit.
#
# What it does:
#   1. Clones Apollo's deception-detection repo at the pinned commit.
#   2. Installs Apollo editable into the project's uv venv with --no-deps.
#      (Apollo pins torch<2.3 which would conflict with our torch>=2.4; --no-deps
#      lets us register the package without disturbing the resolved environment.)
#   3. Authenticates with HuggingFace + W&B from .env (if present).
#   4. Verifies `import deception_detection` works in the project venv.
#
# Required env vars (typically in .env at the project root):
#   HF_TOKEN         — HuggingFace token; Llama-3.3 is gated.
#   WANDB_API_KEY    — Weights & Biases token.
#
# Pinned Apollo commit lives in pyproject.toml. If you change it there, update
# APOLLO_COMMIT below too.

set -euo pipefail

# Force uv to copy package files instead of hardlinking. On RunPod, uv's cache
# lives on container disk and .venv lives on the /workspace volume — different
# filesystems mean hardlinks fail, and uv's fallback can silently leave some
# packages with metadata registered but no actual files on disk (libcudnn.so
# missing was the symptom that uncovered this). Setting copy mode here protects
# the uv commands this script invokes; uv sync should also be run with this
# variable set (see step below).
export UV_LINK_MODE=copy

APOLLO_COMMIT="f8ec4010e74927394709dffa22b97bdf8cd5a62f"
APOLLO_DIR="/workspace/deception-detection"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"

# Put the venv on container-local disk, not the /workspace network volume.
# RunPod's /workspace is MooseFS-backed and intermittently throws stale file
# handle errors (errno 116) under the parallel writes uv does during a sync
# of our 200+ package tree. Container-local /root is fast and reliable; the
# trade-off is that .venv is lost on pod restart, but `uv sync` from cache
# rebuilds in ~30s, and code + data persist via git and HF Hub anyway.
#
# Respects a pre-set UV_PROJECT_ENVIRONMENT so an operator can override
# (e.g. for an unusual pod layout where /root isn't local).
if [ -z "${UV_PROJECT_ENVIRONMENT:-}" ]; then
    export UV_PROJECT_ENVIRONMENT="/root/.venvs/deception-probe-alignment-faking"
fi
mkdir -p "$(dirname "$UV_PROJECT_ENVIRONMENT")"
echo "==> Project venv: $UV_PROJECT_ENVIRONMENT"

# Persist the venv path for future shells on this pod. Idempotent — only
# appends once. Future `uv run` calls in a fresh shell will pick the right
# venv without the operator having to re-export anything.
#
# Fresh RunPod containers often don't ship a /root/.bashrc, so the old
# `[ -f /root/.bashrc ]` guard would silently skip the append on the very
# pods that need it most. Touch the file first to make the append
# unconditional on the existence check, then dedupe via marker grep.
touch /root/.bashrc
BASHRC_MARKER="# >>> deception-probe-alignment-faking venv path (auto-added by 00_runpod_setup.sh) <<<"
if ! grep -qF "$BASHRC_MARKER" /root/.bashrc; then
    {
        echo ""
        echo "$BASHRC_MARKER"
        echo "export UV_PROJECT_ENVIRONMENT=\"$UV_PROJECT_ENVIRONMENT\""
        echo "export UV_LINK_MODE=copy"
    } >> /root/.bashrc
    echo "==> Persisted UV_PROJECT_ENVIRONMENT + UV_LINK_MODE to /root/.bashrc for future shells"
else
    echo "==> /root/.bashrc already carries UV_PROJECT_ENVIRONMENT; no change."
fi

if [ ! -f "$PROJECT_DIR/pyproject.toml" ]; then
    echo "ERROR: run this from the project root (no pyproject.toml at $PROJECT_DIR)" >&2
    exit 1
fi

echo "==> Project root: $PROJECT_DIR"
echo "==> Apollo target: $APOLLO_DIR @ $APOLLO_COMMIT"

echo "==> Syncing project venv (uv sync with UV_LINK_MODE=copy)"
uv sync

if [ -d "$APOLLO_DIR/.git" ]; then
    echo "==> Apollo directory exists; updating to pinned commit"
    git -C "$APOLLO_DIR" fetch --quiet origin
    git -C "$APOLLO_DIR" checkout --quiet "$APOLLO_COMMIT"
else
    echo "==> Cloning Apollo"
    git clone https://github.com/ApolloResearch/deception-detection "$APOLLO_DIR"
    git -C "$APOLLO_DIR" checkout --quiet "$APOLLO_COMMIT"
fi

echo "==> Installing Apollo editable with --no-deps (avoids torch downgrade)"
# `uv pip install` doesn't follow UV_PROJECT_ENVIRONMENT (that's only consulted
# by project-based `uv sync` / `uv run`). Pass --python explicitly so the
# editable install lands in the venv we just synced, not in the system Python.
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-deps -e "$APOLLO_DIR"

echo "==> Authenticating HuggingFace + W&B from .env"
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.env"
    set +a

    if [ -n "${HF_TOKEN:-}" ]; then
        uv run huggingface-cli login --token "$HF_TOKEN" \
            || echo "  WARNING: HF login failed (check HF_TOKEN)"
    else
        echo "  HF_TOKEN not set in .env; skipping HF login"
    fi

    if [ -n "${WANDB_API_KEY:-}" ]; then
        uv run wandb login "$WANDB_API_KEY" \
            || echo "  WARNING: W&B login failed (check WANDB_API_KEY)"
    else
        echo "  WANDB_API_KEY not set in .env; skipping W&B login"
    fi
else
    echo "  No .env at $PROJECT_DIR/.env; skipping auth"
fi

echo "==> Verifying Apollo import"
uv run python -c "import deception_detection; print(f'  Apollo OK: {deception_detection.__file__}')"

echo "==> Verifying GPU access"
uv run python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

echo "==> Setup complete."
