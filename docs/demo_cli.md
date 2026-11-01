# Generate stories with the trained models

This demo continues a story opening using a saved model. You can try one model
or compare several models using exactly the same opening.

It does not train anything or change checkpoints. It is a story continuation
demo, not a chatbot: each new opening starts a separate story.

## Before you start

Open a terminal **in the project root**, where `demo.py` is located.
Activate the project's Python environment:

```bash
source .venv/bin/activate
```

The checkpoints and local tokenizer must already be present. This workspace has
them. The demo does not download files or require W&B, internet access, or the
training stories. If using another computer, copy the checkpoints and the
`data/full_tokens/tokenizer/` folder too, and install `requirements.txt`.

## Easiest option: interactive demo

```bash
python demo.py
```

1. Choose a model with **Up/Down**, then press **Enter**. You can also type its
   number or name. The initial selection is `e-adjusted`.
2. Enter the beginning of a story, for example:

   ```text
   Once upon a time, there was a little bear named Teddy. Teddy
   ```

3. Wait for the continuation. Diffusion models are slower than the baseline.
4. Enter another story opening to try again.
5. Type `/quit` to finish. Ctrl+C also exits.

Type `/models` to choose another model without leaving the demo. To skip the model menu:

```bash
python demo.py --model e-baseline
```

The interface has a compact session card, an active-model summary, a prompt area
and a separate response panel for each model. Model comparisons show their
position (1/2, 2/2), and loading/generation indicators show elapsed time. The
story text wraps when you resize the terminal.

| Control | What it does |
| --- | --- |
| Up/Down + Enter in the model picker | Select a checkpoint |
| Number or model name + Enter | Select a checkpoint by typing |
| Esc in the model picker | Cancel; `/models` keeps the current model |
| `/models` at the story prompt | Switch checkpoints without restarting |
| `/help` | Show the available shortcuts |
| Up/Down at the story prompt | Recall earlier openings, when line editing is available |
| `/quit` or Ctrl+C | Finish the demo |

Arrow-key selection is available in supported POSIX colour terminals. Plain,
small, or unsupported terminals use the numbered/name menu. Input history stays
in memory; the demo does not save a history file. Redirected output has no colour
or animation.

For a text-only display:

```bash
python demo.py --plain
```

The `NO_COLOR` environment variable also disables colour and animations.

## Available models

```bash
python demo.py --list
```

`ready` means the checkpoint file exists; `missing` means it must be copied to
the listed location, or its location must be updated in `configs/demo.yaml`.

| Name | Experiment | Model |
| --- | --- | --- |
| `d-baseline` | D: 12 layers, trained on 50,000 stories | Autoregressive baseline |
| `d-original` | D: 12 layers, trained on 50,000 stories | Diffusion with original noise |
| `d-adjusted` | D: 12 layers, trained on 50,000 stories | Diffusion with adjusted noise |
| `e-baseline` | E: 30 layers, trained on 200,000 stories | Autoregressive baseline |
| `e-original` | E: 30 layers, trained on 200,000 stories | Diffusion with original noise |
| `e-adjusted` | E: 30 layers, trained on 200,000 stories | Diffusion with adjusted noise |

D uses the uploaded final checkpoints in `runs/Archive/`. E uses the
best-validation checkpoints in the `runs/full_*_seed42_dataset/` directories.

## Generate once and exit

Put the opening in quotes so it is treated as one argument:

```bash
python demo.py --model e-adjusted --prompt "Once upon a time, there was a little bear named Teddy. Teddy"
```

The terminal shows the original **Prompt** and the generated **Continuation**
separately. For a baseline example, replace `e-adjusted` with `e-baseline`.
If `--model` is omitted, the demo uses `e-adjusted`.

## Compare models on the same opening

Give several model names after `--model`:

```bash
python demo.py --model d-adjusted e-adjusted --prompt "Once upon a time, there was a little bear named Teddy. Teddy"
```

To compare all three E variants:

```bash
python demo.py --model e-baseline e-original e-adjusted --prompt "One day, Lily found a shiny key."
```

Results appear one after another, labelled by model. All models receive the same
prompt, requested continuation length and generation seed. Only one model is
kept loaded at a time. Each additional diffusion model adds waiting time.

## Shorter examples and different seeds

```bash
python demo.py --model e-adjusted --prompt "One day, Lily found a shiny key." --tokens 24 --seed 457
```

- `--tokens` controls how many **new tokens** to compute. A token is a piece of
  text, not necessarily a word. The default is 48.
- `--seed` controls diffusion's initial random noise. The default is 456.
  A different seed can produce a different continuation. The baseline uses
  argmax and does not change its output when only the seed changes.
- Repeating a command on the same device and software setup should reproduce
  its text. Different devices or software versions can change results.

The visible continuation ends at the first end-of-story token (EOS), so it can
be shorter than requested or empty. The sampler still computes the requested
number of tokens, matching the evaluation procedure.

## Save examples

Add `--output` to append results to a JSONL file:

```bash
python demo.py --model d-adjusted e-adjusted --prompt "One day, Lily found a shiny key." --output runs/demo/examples.jsonl
```

The directory is created automatically. Each line is one model's result, with
its prompt, continuation, token IDs, checkpoint hash, generation settings and
time. Running the command again **adds** lines; it does not replace earlier ones.
The same option works in interactive mode and saves each example as it finishes.

## CPU, GPU and other checkpoints

The demo automatically uses CUDA when available, otherwise CPU. CPU generation
works but can be slow, especially for diffusion. You can choose explicitly:

```bash
python demo.py --model e-baseline --device cpu --prompt "One day, Lily found a shiny key." --tokens 16
```

To load another compatible checkpoint from this project:

```bash
python demo.py --checkpoint runs/Archive/prueba_d_diffusion_noise_shifted/checkpoint.pt --prompt "One day, Lily found a shiny key."
```

Use `--model` or `--checkpoint`, not both. Paths must be relative to the project
root. The checkpoint must contain this project's saved `config`, `model` and
`metadata`; an arbitrary downloaded model file will not work.

## Settings and common problems

`configs/demo.yaml` contains model paths and generation defaults. To use a copy
of that file, pass `--config configs/my_demo.yaml`.

The default demo matches the current evaluation: FP32, argmax decoding, eight
Euler steps per diffusion block, clamping and final-denoiser decoding. Each
checkpoint keeps its own original or adjusted noise intervals.

- **Checkpoint not found:** run `python demo.py --list`, then check the paths in
  `configs/demo.yaml`.
- **Tokenizer missing or does not match:** restore the original local tokenizer
  folder. Its file hashes must match those recorded in the checkpoint.
- **Prompt is too long:** shorten it or lower `--tokens`. The default demo limit
  is 256 tokens for the opening and continuation combined; text is not silently
  truncated.
- **CUDA unavailable:** use `--device cpu`, or use the environment where the GPU
  training and evaluation ran.
- **Text contains mistakes:** these are experimental models. The demo displays
  their generated text; it does not rewrite or correct it.

For every option, run `python demo.py --help`. This CLI evaluates generation
examples only. For held-out losses and the saved evaluation tables, use
`scripts/evaluate_full.py` and the commands in [full_training.md](full_training.md).

Implementation check: all six real D/E checkpoints reproduced the exact 48 token
IDs from the existing FP32 evaluation for its first prompt and seed. The local
verification is in `runs/demo_validation/verification.json`. The project test
suite passes 40 tests, including interactive use, comparison/export and EOS handling.
