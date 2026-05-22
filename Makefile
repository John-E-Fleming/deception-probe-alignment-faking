# HuggingFace Hub data management
# Requires: HF_TOKEN in .env, huggingface-cli installed (via uv)
# Data is organized per-experiment: data/experiments/<model_name>/

HF_DATA_REPO := John-E-Fleming/probe-control-arena-data

.PHONY: hf-login hf-pull-data hf-push-data

hf-login:
	uv run huggingface-cli login --token $$(grep HF_TOKEN .env | cut -d= -f2 | tr -d ' "')

hf-pull-data:
	uv run huggingface-cli download $(HF_DATA_REPO) --repo-type dataset --local-dir data/

hf-push-data:
	uv run huggingface-cli upload $(HF_DATA_REPO) data/ . --repo-type dataset
