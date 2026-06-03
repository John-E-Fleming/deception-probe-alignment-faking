# HuggingFace Hub data management.
#
# Requires: HF_TOKEN in .env, huggingface-cli installed (via uv sync).
#
# The repo is PRIVATE while the sprint is in progress. Review whether to
# make it public at sprint close, once the final dataset is settled and
# any sensitive content (raw harmful queries, model rollouts) has been
# audited.
#
# data/redwood_honly/ is vendored (MIT-licensed reproduction of Redwood's
# helpful-only prompts + classifier examples) and lives in git, not on
# the Hub — `hf-push-data` excludes it to avoid duplicating tracked files
# into Hub artifacts.

HF_DATA_REPO := John-E-Fleming/deception-probe-alignment-faking-data

# Pin the uv venv path so `uv run` doesn't fall back to creating a fresh
# .venv in the project directory when the user's shell predates the
# bashrc append from 00_runpod_setup.sh. The fallback would install our
# project deps but NOT Apollo (Apollo is installed editable + --no-deps
# in a separate setup step), so subsequent `uv run python …` calls hit
# ModuleNotFoundError: No module named 'deception_detection'. Honour a
# pre-set value; on RunPod the setup script default is
# /root/.venvs/deception-probe-alignment-faking; for local dev fall back
# to the in-project .venv.
ifndef UV_PROJECT_ENVIRONMENT
  ifneq (,$(wildcard /root/.venvs/deception-probe-alignment-faking))
    UV_PROJECT_ENVIRONMENT := /root/.venvs/deception-probe-alignment-faking
  else
    UV_PROJECT_ENVIRONMENT := $(CURDIR)/.venv
  endif
endif
export UV_PROJECT_ENVIRONMENT
export UV_LINK_MODE := copy

.PHONY: hf-login hf-pull-data hf-push-data

hf-login:
	uv run huggingface-cli login --token $$(grep HF_TOKEN .env | cut -d= -f2 | tr -d ' "')

hf-pull-data:
	uv run huggingface-cli download $(HF_DATA_REPO) --repo-type dataset --local-dir data/

hf-push-data:
	uv run huggingface-cli upload $(HF_DATA_REPO) data/ . --repo-type dataset --exclude "redwood_honly/*"
