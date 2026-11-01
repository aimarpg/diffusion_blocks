# Full 30-layer training experiment

This experiment trains from scratch with the SmolLM2-135M dimensions: 30 layers,
width 576, MLP 1536, 9 attention heads, 3 KV heads and vocabulary 49152.
The baseline has one 30-layer stack. Each diffusion model has three independent
10-layer blocks. Their adapters add parameters: baseline has 134,515,008 total
parameters and diffusion has 144,628,992. The random, tied embedding/output table
is frozen in all variants. No pretrained SmolLM2 weights are imported.

The three configs are `configs/full_baseline.yaml`,
`configs/full_diffusion_original.yaml`, and `configs/full_diffusion_adjusted.yaml`.
The diffusion noise configurations are preserved from D. Original noise is an
ablation: two blocks previously had zero loss without learning. E records this
as `inactive_blocks`; it does not abort the experiment for this expected finding.
Existing probes A/B/C/D are unchanged.

## 1. Environment and offline GPU verification

Run commands from the repository root in a CUDA environment. Python 3.11 is the
recommended installation; Python 3.12 also ran the verification in this workspace.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -m pytest tests/ -q
python run.py --smoke
python scripts/train_full.py --phase verify
```

`verify` uses the authored byte-token fixture, the full model dimensions, FP16,
sequence length 256, microbatch 2 and accumulation 8. It performs six optimizer
attempts per model (two per diffusion block), logs to W&B offline, and checks GPU
execution/checkpointing without downloading TinyStories. It is not a language
quality measurement. Its artifacts are `runs/verify_full_*_seed42/`.

If a phase was interrupted, repeat its command with `--resume`. New runs refuse
to overwrite a nonempty output directory. The verification already run in this
workspace has artifacts; use `--resume` to revisit those outputs.

## 2. Prepare the shared data once

```bash
python scripts/prepare_pilot.py --config configs/prepare_full.yaml
```

Preparation downloads/tokenizes 200,000 unique official training stories and
4,000 unique official validation stories. The first 2,000 validation stories are
used during training; the next 2,000 are reserved as local `test`. All partitions
are deduplicated against previously selected partitions. This is an explicit
local held-out partition of official validation, not an official TinyStories test
split. It excludes the first 500 validation stories used in D.

`data/full_tokens/manifest.json` records resolved dataset/tokenizer commit SHAs,
counts and SHA256 hashes, including held-out data and tokenizer files. All methods
use this same cache. Keep the cache/manifest to reproduce the exact data; preparation
refuses an existing manifest. No frozen splits module is changed.

## 3. Calibrate with real data

```bash
python scripts/train_full.py --phase calibrate
```

Runs the three models sequentially, at the training batch/context settings, for
up to 42 updates or 240 training seconds each. Reports are in
`runs/calibrate_full_*_seed42/report.json`. Inspect `profile`,
`training_tokens_per_second`, memory peaks, `skipped_updates`, initial block
losses and actual `updates_per_block`. Profiling excludes the first 12 updates;
allocated/reserved peaks include training and validation after initial evaluation.
Initial evaluation and checkpoint writes are not optimizer throughput.

The effective batch is 2 sequences × 8 accumulated microbatches = 16 sequences per
optimizer attempt, except a partial batch at an epoch boundary. Gradients are
weighted by valid token counts, so padding and partial batches do not distort the
objective. An attempt selects one diffusion block; the next attempt selects the
next block. Initial FP16 loss scale is configurable and defaults to 1024.

If memory is tight, lower microbatch and increase accumulation proportionally in
all three configs. Keep the same effective batch. Rerun calibration under a new
output name (change the experiment name in the configs), because it affects
throughput. The current `profile_warmup_steps` is only a profiling exclusion;
`lr_warmup_updates` implements actual LR warmup, independently per optimizer.

## 4. Train all three variants

Course-budget run, first seed:

```bash
python scripts/train_full.py --phase train
```

This uses up to 30,000 optimizer attempts or 900 cumulative training-phase seconds
per model, whichever comes first. Validation and periodic checkpoint writes count
against that training time. Setup/initial evaluation and finalization are additional
costs. `presupuesto.total_seconds` is a reported 1,200-second pipeline criterion,
not a watchdog or a guarantee of T4 compliance. W&B shutdown and process imports
also require wall-clock measurement. This workstation is not a T4.

For a longer exploratory run on the workstation:

```bash
python scripts/train_full.py --phase train --budget extended
```

The explicit extended preset allows 3,600 training seconds per model and reports
against a 4,500-second total criterion. **This preset exceeds the course 20-minute
budget** and uses separate `*_extended` output directories. It starts a separate
experiment from scratch; it does not automatically continue course-budget weights.
Budget values and seeds are in `configs/full_experiment.yaml`.

Before the main run, use calibration throughput to choose a suitable `max_steps`
in each model config. The default 30,000 is a ceiling and LR scheduling horizon,
not a promise that it will be reached. LR warms up for 100 successful updates per
optimizer and decays by cosine toward 10% of its initial value over that horizon.
With a time-limited run, the schedule may stop before reaching its minimum. If you
choose different step ceilings for equal-time runs, record those choices.

For three training seeds (42, 43, 44):

```bash
python scripts/train_full.py --phase train --budget extended --all-seeds
```

This launches nine runs sequentially on one GPU. Use the first seed to establish
that the setup works before spending the larger budget. Training uses W&B offline
by default; offline runs are real logged experiments. To upload them:

```bash
wandb login
wandb sync wandb/offline-run-*
```

For online logging, set `wandb.modo: online` in the model configs before starting.

## 5. Resume safely

Repeat the same launcher command with `--resume`:

```bash
python scripts/train_full.py --phase train --budget extended --resume
python scripts/train_full.py --phase train --budget extended --all-seeds --resume
```

Existing checkpoints continue; variants without checkpoints start normally. Runs
already at the step/time ceiling perform no further optimizer updates. Configs,
seed, architecture, data hashes, GPU model and PyTorch version are checked before
continuation. Exact cross-machine or cross-version reproducibility is not promised.
Each resumed invocation opens a W&B run with the same experiment name and continues
global step numbering; local `steps.jsonl` is reconciled to the committed checkpoint.

A checkpoint restores model, AdamW states, scaler, successful update/token counts,
next active block, RNG, current epoch permutation/cursor and the generator for the
next epoch. LR is derived from successful update counts. Mid-update interruption
returns to the last committed optimizer boundary; uncheckpointed work is repeated.
`num_workers` must remain zero for this stream implementation.

To give the *same* run more time after reaching its time cap, increase
`max_train_seconds` and `total_seconds` in the selected suite budget and resume with
the same phase/budget/seed. Keep `max_steps` and all scientific settings unchanged;
changing the scheduling horizon is deliberately rejected on resume.

A standalone config can also use the normal entry point:

```bash
python run.py --config configs/full_baseline.yaml
python run.py --config configs/full_baseline.yaml --resume runs/full_baseline/last.pt
```

Standalone outputs differ from suite outputs (`runs/full_baseline` versus
`runs/full_baseline_seed42`). Use the suite consistently for the default evaluation.

## 6. Compare held-out losses and generated text

Start with the quick evaluation of the corresponding trained preset:

```bash
python scripts/evaluate_full.py --quick
python scripts/evaluate_full.py --budget extended --quick
```

Choose the command matching the training budget. Quick mode uses ten held-out
story prefixes, 48 continuation tokens, one generation seed and 16 test batches.
It is a preview, not the final comparison.

Full evaluation:

```bash
python scripts/evaluate_full.py
python scripts/evaluate_full.py --budget extended
```

Full evaluation uses 100 fixed prefixes from different reserved stories, 128 new
tokens and three generation seeds per diffusion model. The baseline is greedy and
is generated once per prompt, because its output is seed-independent. Diffusion
is stochastic through its initial noise; it uses eight Euler steps per block,
clamping and final-denoiser decoding. The sampler currently computes a fixed token
budget; visible continuations are truncated at the first EOS and that position is
recorded. Latency counts all computed tokens, including tokens beyond EOS.

To evaluate all training seeds, add `--all-seeds`. To resume an interrupted
evaluation without regenerating completed samples, repeat the exact command with
`--resume`. Checkpoint SHA256 checks prevent mixing generations from changed
weights. Full diffusion evaluation can take substantially longer than training:
there is no KV cache and each token requires multiple denoiser passes. Evaluation
has its own budget and is outside the training budget.

Default outputs are in `runs/full_evaluation/`, with `_extended`, `_all_seeds` and
`_quick` suffixes when selected:

| Artifact | Content |
| --- | --- |
| `summary.json` | Per-model held-out metrics, repetition, empty outputs and generation throughput |
| `generations.jsonl` | Prompt, continuation, raw token IDs, model, seed, timing and first EOS |
| `blind_ratings.csv` | Anonymous, deterministically shuffled samples and blank human ratings |
| `blind_key.json` | Mapping from anonymous IDs to model and seed; withhold from raters |
| `prompts.json` | Exact shared token prefixes |
| `checkpoint_hashes.json` | Exact checkpoint identities for evaluation/resume |
| `config.json` | Resolved evaluation settings |

Rate coherence, grammar, relevance and story consistency on one agreed scale
before opening the key. Finish/resume generation before entering ratings, since
resuming rebuilds the rating template. Compare the same prompt IDs across models;
multiple diffusion seeds are repeated samples, not independent training runs.

Baseline CE/perplexity is a next-token likelihood metric. Diffusion block losses
are denoising metrics and are not comparable to baseline perplexity or directly
across different noise distributions. `loss_sigma_max` is only an approximate,
optimistic diagnostic, not an equivalent likelihood. Keep blind output ratings
and generation efficiency central to the cross-method comparison.

## Training artifacts and interpretation

Each run contains `config.json`, `steps.jsonl`, `report.json`, `last.pt` and `best.pt`.
`last.pt` supports resume in its original output directory. `best.pt` is selected
by validation baseline CE or diffusion `loss_sigma_max`, respectively, and is for
inference/evaluation. Both contain full optimizer state and can consume several GB
per run; reserve disk space for all seeds. Model weights in Git remain ignored.

Report quality at the same time budget alongside tokens and updates per block.
Equal attempts are not equal layer updates: one baseline update trains 30 layers,
whereas one diffusion update trains 10. The default experiment uses equal maximum
training-phase time but can also stop at the step ceiling. `report.json` records
actual exposure, completion and timing, rather than claiming convergence.

The experiment preserves the frozen codebook and original/adjusted schedules.
Changing embeddings or choosing a better schedule after inspecting held-out output
would require a new experiment, with validation used for tuning and test reserved
for final evaluation. Scientific interpretation and the course discussion remain
work for the project authors.

## Implementation verification (1 November 2026)

On RTX 5000 Ada 32 GB, Python 3.12 and pinned PyTorch 2.5.1:

- 26 tests passed, including exact interrupted/uninterrupted CPU and CUDA FP16
  training for both methods, token-weighted accumulation, reserved-data integrity,
  evaluation resume and recovery from a torn training log.
- Existing CPU `--smoke` passed; reproducibility checker scored 97/100.
- All three 30-layer variants ran six FP16 optimizer attempts at sequence length
  256 and effective batch 16 using offline fixture data. Baseline applied six
  updates; both diffusion variants applied two updates to each block. None skipped
  updates. Peak allocated memory was approximately 3.0 GB baseline and 2.7 GB
  diffusion. These figures do not replace calibration on real tokenized data.
- Full TinyStories cache preparation and longer E training were not launched.
  Full-architecture T4 course-budget compliance remains unmeasured.

## Train through all 200,000 stories

The `extended` preset has a one-hour training cap **per variant** (roughly three
hours across the three sequential runs, plus setup/finalization), and also stops
at its configured step ceiling. It does not guarantee dataset coverage.

To process every training window once per model, use the dataset preset:

```bash
python scripts/train_full.py --phase train --budget dataset
python scripts/train_full.py --phase train --budget dataset --resume
python scripts/evaluate_full.py --budget dataset --quick
python scripts/evaluate_full.py --budget dataset
```

The second command is only needed to resume an interruption. This preset has no
time cutoff. It derives the number of updates from the actual tokenized windows,
microbatch size and accumulation, overriding the ordinary `max_steps` ceiling.
Every window from the selected 200,000 training stories is visited once; the last
partial accumulation ends at the epoch boundary instead of repeating examples.
The LR horizon uses this derived update count. `report.json` includes
`training_windows`, `target_steps` and `epochs_completed`.

Diffusion retains round-robin block training: the model collectively processes
one full dataset pass, and each block receives about one-third of its batches.
This is equal dataset exposure per model, not a complete pass per block. FP16
skipped optimizer updates remain reported separately from data coverage.

`configs/full_experiment.yaml` sets `budgets.dataset.epochs: 1`. Increase that value
**before starting** for additional passes; changing it on resume is rejected since
it changes the LR horizon. Add `--all-seeds` for the three configured training
seeds. Outputs use the `_dataset` suffix, separate from course/extended runs.

Actual duration depends on the number of tokenized windows and measured speed;
200,000 stories do not mean 200,000 optimizer updates. Use real-data calibration
to estimate runtime. This mode can exceed the course's 20-minute requirement and
is for the requested full-dataset experiment.

Epoch-mode verification: 28 tests pass, including full window coverage without
repetition, partial final batches and exact resume for both model methods.

## Compare microbatch 32 versus 48

```bash
python scripts/benchmark_batches.py
```

This runs six calibrations sequentially: all three variants at microbatch 32,
then all three at 48, using accumulation 1. Candidates are declared in
`configs/batch_calibration.yaml`; training YAMLs and previous calibration outputs
are not modified. Each child process starts with a fresh CUDA context, so a failed
or out-of-memory candidate does not contaminate the following measurements.
Each run uses the regular calibration budget (42 update attempts / 240 seconds),
excluding the first 12 attempts from reported throughput.

The final table shows tokens/second, allocated/reserved GiB and skipped updates.
Results and checkpoints are under `runs/batch_calibration/`, with a combined
`comparison.json`. A failed candidate is marked `failed`; incomplete measurement
or skipped updates are marked `check` and should not be declared the winner.
Compare batch sizes within the same model variant. Throughput is computed from
valid tokens; effective batches differ (32 versus 48), so this is a performance
benchmark, not evidence of equivalent learning quality.

```bash
python scripts/benchmark_batches.py --report-only
python scripts/benchmark_batches.py --resume
python scripts/benchmark_batches.py --dry-run
```

The first command redisplays recorded results; the second continues interrupted
runs and skips completed ones; the third previews the six configurations without
training. A candidate that repeatedly fails with OOM needs a smaller batch, not
repeated resumes. After selecting a batch that fits and performs well across all
three models, set its `datos.batch_size` and
`entrenamiento.gradient_accumulation_steps: 1` in all three main model configs
before starting full training. The benchmark does not change these settings.
