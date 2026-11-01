# Downloadable D/E models

The `models-v1` GitHub release distributes the six original checkpoints and the
matching local tokenizer. Weights stay outside Git history; keep `runs/`,
`data/*` and `*.pt` in `.gitignore`.

## Download on another computer

Run from the repository root after installing `requirements.txt`:

```bash
python scripts/download_models.py --list
python scripts/download_models.py --models e-adjusted
python demo.py --model e-adjusted
```

Omit `--models` to download all six models (about 5.95 GiB plus the tokenizer).
Multiple names are accepted, for example `--models d-adjusted e-adjusted`.
The tokenizer is included automatically. The script uses the standard library,
streams files to disk and verifies their exact size and SHA-256 before installing
them at the paths in `configs/demo.yaml`. Matching local files are verified and
skipped. Files with different bytes require explicit `--force`; failed downloads
leave the old file intact. Interrupted downloads restart on the next invocation.
Allow space for the selected models and one temporary download. The release must
be published and accessible without authentication for this downloader to work.

Model names, destinations, sizes and hashes are recorded in
`configs/model_assets.json`. A fresh clone does not contain the checkpoint files.
The CLI itself remains offline after this one-time download.

## Maintainer: prepare and publish

With the six trained checkpoints and tokenizer already present:

```bash
python scripts/prepare_model_release.py
```

Settings live in `configs/model_release.yaml`. Preparation creates named assets
in `runs/model_release/` and writes `configs/model_assets.json`. It uses hard links
when supported, otherwise copies files. Do not edit the prepared `.pt` assets:
hard-linked files share storage with the originals. No weights, optimizer state,
configuration or precision are changed. Freeze these source checkpoints while
preparing and uploading. The complete originals support the existing CLI and
evaluators, and retain training-resume state.

1. Commit and push the CLI, download scripts, manifest and documentation first.
   Keep all binary assets ignored.
2. In GitHub, open **Releases → Draft a new release**. Create tag `models-v1`
   targeting the pushed `feature/full-dataset-training` commit containing the CLI.
3. Use `docs/model_release_notes.md` for the release description.
4. Attach **every file** in `runs/model_release/`: six distinct `.pt` files,
   five `tokenizer-*` files, and `model_assets.json`. Upload them individually;
   a single combined archive would exceed the per-asset limit.
5. Publish the release. The downloader cannot access an unpublished draft.
6. From a fresh clone, download one model and run a short CLI generation.

If GitHub CLI is installed and authenticated, steps 2–5 can instead use:

```bash
gh release create models-v1 runs/model_release/* \
  --repo aimarpg/diffusion_blocks \
  --target feature/full-dataset-training \
  --title "D/E model checkpoints v1" \
  --notes-file docs/model_release_notes.md
```

Do not replace published assets under the same tag. For another version, change
the tag in `configs/model_release.yaml`, use a fresh output directory, regenerate
the manifest and publish another release.

## Comparison reports

`reports/model_comparison/` contains byte-for-byte copies of the existing quick
FP32 evaluation evidence for D/E, including matched generations. Its README lists
the original sources. These are preliminary comparisons with different model
depths, training sizes and schedules; diffusion denoising loss is not perplexity.

Downloading checkpoints and tokenizer is sufficient for CLI generation and
comparing continuations. Recomputing held-out metrics also needs the prepared
TinyStories cache; follow `docs/full_training.md`. The release does not include
the training dataset.
