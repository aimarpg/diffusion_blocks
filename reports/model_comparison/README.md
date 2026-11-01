# Preliminary D/E comparison evidence

These files are byte-for-byte copies of the existing quick FP32 results in:

- `runs/probe_d_vs_full_dataset_quick/`: comparison tables and paired generations.
- `runs/probe_d_evaluation_fp32_quick/`: D evaluation settings, hashes and outputs.
- `runs/full_evaluation_fp32_dataset_quick/`: E evaluation settings, hashes and outputs.

Start with `probe_d_vs_full_dataset_quick/comparison.csv` or
`probe_d_vs_full_dataset_quick/matched_generations.csv`.

All six checkpoint hashes in these evaluations match `configs/model_assets.json`.
The original `runs/` references inside reports are retained for provenance;
the corresponding report files are copied here under the same directory names.
These copies can be inspected without downloading checkpoints or preparing data.
Blind-rating sheets are templates, not completed human quality assessments.

Both evaluations use the same quick settings and prompt IDs: FP32, ten prompts,
48 generated tokens, a single generation seed, and a limited number of held-out
batches. D and E differ in model depth, training data size, checkpoint selection
and training schedules. These results do not isolate any one of those factors.
Diffusion `loss_sigma_max` is a denoising diagnostic, not next-token likelihood
or perplexity. Read the limitations in `comparison.json` before interpreting
the reported changes. No new training or evaluation was performed for this copy.
