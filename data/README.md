# Data Directory

Local storage for experiment artifacts. Subdirectories under `data/` (other
than `redwood_honly/`) are gitignored and persisted remotely via the
project's private HuggingFace Hub dataset repo.

- **HF Hub repo:** `John-E-Fleming/deception-probe-alignment-faking-data`
  (private during sprint; review for public release at sprint close)

## Directory Structure

| Subdirectory     | Contents                                                 | Persistence | Typical Size |
|------------------|----------------------------------------------------------|-------------|--------------|
| `raw/`           | Generated AF transcripts (JSONL — prompt, scratchpad, response, metadata) | HF Hub      | ~1-10 MB     |
| `labelled/`      | Transcripts + AF classifier scores (JSONL)               | HF Hub      | ~1-10 MB     |
| `activations/`   | Per-layer per-token activation tensors (.npz) keyed by sample_id | HF Hub      | ~100 MB - 5 GB |
| `probes/`        | Trained probe weights bundled with scaler (.pkl)         | HF Hub      | ~1 MB        |
| `redwood_honly/` | Vendored MIT-licensed Redwood helpful-only prompts + AF classifier examples | **git** (not HF) | ~50 KB |

`results/` (one level up from `data/`) holds final committed outputs —
transfer matrix JSON, ROC curves, `experiment_metadata.json`. Those stay in
git and don't sync through the Hub.

## Sync Commands

```bash
make hf-login       # Authenticate with HF Hub (reads HF_TOKEN from .env)
make hf-pull-data   # Download data/ from HF Hub
make hf-push-data   # Upload data/ to HF Hub (excludes redwood_honly/)
```

After every RunPod pod restart, run `make hf-pull-data` to restore any
activations / transcripts you had previously generated, rather than
re-running expensive extraction.

## Provenance

The data classes embed provenance inline rather than via separate metadata
files:
- `Transcript.metadata` records model, quantisation, temperature,
  `max_new_tokens`, input index, and tier flag per rollout.
- `LabelledTranscript` adds classifier confidence and inclusion flag.
- `ActivationDataset.metadata` records Apollo commit hash, detection
  mask kind, extraction date, layer, and model.
- `ProbeResult.metadata` records the source scenario, training AUROC,
  n_train, n_positive, and any additional notes.

If the project grows to need separate YAML provenance files per
subdirectory (as in the `faithfulness-probe-control-arena` project), this
is the natural place to add them.
