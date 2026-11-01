# D/E model checkpoints v1

Six trained TinyStories checkpoints for this repository's CLI and evaluation:

- D: baseline, original-noise diffusion, adjusted-noise diffusion;
  12 layers, 50k training stories, final checkpoints.
- E: the same three variants; 30 layers, 200k training stories,
  best-validation checkpoints.

The assets include the matching tokenizer and a SHA-256 manifest. Checkpoints
are the original files, including their saved configuration and metadata.

From a clone of this release with dependencies installed:

```bash
python scripts/download_models.py --models e-adjusted
python demo.py --model e-adjusted
```

Omit `--models` to download all six. See `docs/model_release.md` for setup and
`reports/model_comparison/` for preliminary quick FP32 comparison results.
Generation runs locally after downloading. Recomputing held-out metrics requires
the separate TinyStories cache. These are experimental story-continuation models.
