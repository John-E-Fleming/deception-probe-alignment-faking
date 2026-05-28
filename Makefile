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

.PHONY: hf-login hf-pull-data hf-push-data

hf-login:
	uv run huggingface-cli login --token $$(grep HF_TOKEN .env | cut -d= -f2 | tr -d ' "')

hf-pull-data:
	uv run huggingface-cli download $(HF_DATA_REPO) --repo-type dataset --local-dir data/

hf-push-data:
	uv run huggingface-cli upload $(HF_DATA_REPO) data/ . --repo-type dataset --exclude "redwood_honly/*"
