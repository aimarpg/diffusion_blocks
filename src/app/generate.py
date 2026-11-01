"""Standalone checkpoint inference and terminal interface; no training imports."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import torch
import yaml

from src.app.terminal import ModelSelectionCancelled, Terminal
from src.data.text import ByteTokenizer, TextTokenizer
from src.models.factory import construir_modelo
from src.utils.seed import seed_temporal


def local_path(value: str | Path) -> Path:
    """Accept project-relative paths without parent-directory traversal."""
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Use a path relative to the project root, without "..".')
    return path


def select_device(name: str) -> torch.device:
    """Select CUDA when available, or CPU when explicitly requested or needed."""
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name not in ('cpu', 'cuda'):
        raise ValueError('Device must be auto, cpu or cuda.')
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable. Use --device cpu.')
    return torch.device(name)


def _tokenizer(cfg: dict, metadata: dict, tokenizer_path: Path) -> TextTokenizer:
    if cfg['datos']['source'] == 'fixture':
        return ByteTokenizer()
    from transformers import AutoTokenizer

    if metadata['vocab_size'] != cfg['modelo']['vocab_size'] or metadata['eos_id'] != cfg['datos']['eos_id']:
        raise ValueError('Checkpoint tokenizer metadata is incompatible with the model.')
    for name, expected in metadata['tokenizer_hashes'].items():
        path = tokenizer_path / local_path(name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Tokenizer file does not match the checkpoint: {path}')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if len(tokenizer) != metadata['vocab_size'] or tokenizer.eos_token_id != metadata['eos_id']:
        raise ValueError('Tokenizer vocabulary or EOS does not match the checkpoint.')
    return tokenizer


class CheckpointGenerator:
    """Load one model and its local tokenizer for repeated FP32 generation."""

    def __init__(self, checkpoint: Path, settings: dict, device: torch.device) -> None:
        self.checkpoint = local_path(checkpoint)
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f'Checkpoint not found: {self.checkpoint}. Run python demo.py --list.')
        self.device = device
        self.settings = copy.deepcopy(settings)
        if settings['precision'] != 'fp32':
            raise ValueError('The demo uses precision: fp32 to match the current evaluation.')
        generation = settings['generation']
        if generation['steps_per_block'] < 1 or generation['max_total_tokens'] < 2:
            raise ValueError('Invalid generation limits in the demo configuration.')
        # mmap avoids reading large saved optimizer tensors into RAM unnecessarily.
        saved = torch.load(self.checkpoint, map_location='cpu', weights_only=True, mmap=True)
        cfg = copy.deepcopy(saved['config'])
        cfg['modelo']['noise'].update(steps_per_block=generation['steps_per_block'], clamp=generation['clamp'])
        self.config = cfg
        self.tokenizer = _tokenizer(cfg, saved['metadata'], local_path(settings['tokenizer_path']))
        with seed_temporal(cfg['experimento']['seed']):
            self.model = construir_modelo(cfg)
        self.model.load_state_dict(saved['model'])
        del saved
        self.model = self.model.to(device).eval()
        with self.checkpoint.open('rb') as file:
            self.checkpoint_sha256 = hashlib.file_digest(file, 'sha256').hexdigest()

    @torch.inference_mode()
    def generate(self, prompt: str, seed: int, max_new_tokens: int) -> dict:
        """Return continuation, raw token IDs and settings using the evaluation sampler."""
        if not prompt.strip():
            raise ValueError('Enter a nonempty story opening.')
        if max_new_tokens < 1:
            raise ValueError('--tokens must be a positive integer.')
        if not 0 <= seed < 2 ** 32:
            raise ValueError('--seed must be between 0 and 4294967295.')
        if isinstance(self.tokenizer, ByteTokenizer):
            ids = self.tokenizer.encode(prompt)
        else:
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not ids:
            raise ValueError('The prompt contains no tokens.')
        limit = self.settings['generation']['max_total_tokens']
        if len(ids) + max_new_tokens > limit:
            raise ValueError(f'Prompt ({len(ids)} tokens) + --tokens ({max_new_tokens}) exceeds '
                             f'the demo limit of {limit}. Shorten the prompt or use fewer tokens.')
        inputs = torch.tensor([ids], device=self.device)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        with seed_temporal(seed):
            generated = self.model.generate(inputs, max_new_tokens)[0, len(ids):].tolist()
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        eos = self.config['datos']['eos_id']
        eos_index = generated.index(eos) if eos in generated else None
        visible = generated if eos_index is None else generated[:eos_index]
        return {'model': self.config['experimento']['nombre'],
                'checkpoint': str(self.checkpoint), 'checkpoint_sha256': self.checkpoint_sha256,
                'method': self.config['modelo']['method'],
                'layers': self.config['modelo']['num_hidden_layers'],
                'device': str(self.device), 'precision': self.settings['precision'],
                'seed': seed, 'max_new_tokens': max_new_tokens,
                'noise': copy.deepcopy(self.config['modelo']['noise']),
                'prompt': prompt, 'prompt_token_ids': ids,
                'continuation': self.tokenizer.decode(visible), 'token_ids': generated,
                'first_eos_index': eos_index, 'generated_tokens': len(generated),
                'seconds': seconds, 'tokens_per_second': len(generated) / seconds}


def list_models(settings: dict, terminal: Terminal | None = None) -> None:
    """Print grouped model aliases, checkpoint availability and descriptions."""
    view = terminal if terminal is not None else Terminal()
    view.heading('Model library', 'D / learning pilot     E / full-dataset experiment')
    available = {name for name, entry in settings['models'].items() if local_path(entry['checkpoint']).is_file()}
    view.catalogue(settings['models'], settings.get('groups', {}), available, paths=True)


def _choose_model(settings: dict, view: Terminal, default: str | None = None) -> str:
    available = {name for name, entry in settings['models'].items() if local_path(entry['checkpoint']).is_file()}
    if not available:
        raise FileNotFoundError('No configured checkpoints are present. Run python demo.py --list.')
    default = default or settings['default_model']
    if default not in available:
        default = next(name for name in settings['models'] if name in available)
    return view.choose_model(settings['models'], settings.get('groups', {}), available, default)


def main(argv: list[str] | None = None) -> int:
    """Run one-shot or interactive generation, optionally appending JSONL results."""
    parser = argparse.ArgumentParser(description='Generate stories with saved D/E models. No training or downloads.')
    parser.add_argument('--config', type=Path, default=Path('configs/demo.yaml'), help='Demo settings YAML.')
    parser.add_argument('--list', action='store_true', help='List model names and checkpoint availability, then exit.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--model', nargs='+', metavar='NAME', help='One or more aliases from --list; evaluated sequentially.')
    group.add_argument('--checkpoint', type=Path, help='Another project checkpoint, using its saved architecture.')
    parser.add_argument('--prompt', help='Story opening; omit to enter prompts interactively.')
    parser.add_argument('--tokens', '--max-new-tokens', dest='max_new_tokens', type=int, help='New tokens per model (default in YAML).')
    parser.add_argument('--seed', type=int, help='Generation seed (default in YAML).')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), help='auto chooses CUDA if available, otherwise CPU.')
    parser.add_argument('--output', type=Path, help='Append one JSON object per model and prompt to this JSONL file.')
    parser.add_argument('--plain', action='store_true', help='Use plain text without colour, animation or Unicode decoration.')
    args = parser.parse_args(argv)
    view = Terminal(plain=args.plain)
    try:
        settings = yaml.safe_load(local_path(args.config).read_text())
        if args.list:
            list_models(settings, view)
            return 0
        seed = args.seed if args.seed is not None else settings['generation']['seed']
        count = args.max_new_tokens if args.max_new_tokens is not None else settings['generation']['max_new_tokens']
        if count < 1 or not 0 <= seed < 2 ** 32:
            raise ValueError('Use a positive --tokens and a --seed between 0 and 4294967295.')
        if args.prompt is not None and not args.prompt.strip():
            raise ValueError('--prompt must contain a story opening.')
        output = local_path(args.output) if args.output else None
        if output is not None and output.suffix != '.jsonl':
            raise ValueError('--output must end in .jsonl; results are appended as JSON lines.')
        device = select_device(args.device or settings['device'])
        torch.set_num_threads(settings['cpu_threads'])
        view.heading('Story studio', 'Choose a model. Begin a story. Compare what follows.')
        if args.checkpoint:
            selected = [('custom', local_path(args.checkpoint))]
        else:
            names = args.model or ([settings['default_model']] if args.prompt is not None else [_choose_model(settings, view)])
            unknown = set(names) - settings['models'].keys()
            if unknown:
                raise ValueError(f'Unknown model: {", ".join(sorted(unknown))}. Run python demo.py --list.')
            selected = [(name, local_path(settings['models'][name]['checkpoint'])) for name in dict.fromkeys(names)]
        for _, checkpoint in selected:
            if not checkpoint.is_file():
                raise FileNotFoundError(f'Checkpoint not found: {checkpoint}. Run python demo.py --list.')
        view.session(str(device), seed, count, [name for name, _ in selected])
        if args.prompt is None:
            view.notice('Write an opening in English. Each opening starts a new story.')
            view.notice('Use /help for shortcuts. Up/Down recalls earlier openings when available.')
        generator = None
        while True:
            prompt = args.prompt if args.prompt is not None else view.ask('Story opening')
            if args.prompt is None and prompt.strip() == '/quit':
                break
            if not prompt.strip():
                view.notice('Enter a story opening, or /quit to exit.')
                continue
            if args.prompt is None and prompt.strip() == '/help':
                view.help()
                continue
            if args.prompt is None and prompt.strip() == '/models':
                try:
                    name = _choose_model(settings, view, default=selected[0][0])
                except ModelSelectionCancelled:
                    view.notice('Keeping the current model.')
                    continue
                selected = [(name, local_path(settings['models'][name]['checkpoint']))]
                view.session(str(device), seed, count, [name])
                continue
            if args.prompt is None and prompt.strip().startswith('/'):
                view.notice('Unknown command. Use /help to see the available shortcuts.')
                continue
            view.prompt(prompt)
            for index, (name, checkpoint) in enumerate(selected, start=1):
                label = settings['models'][name].get('label', settings['models'][name]['description']) if name in settings['models'] else str(checkpoint)
                view.model(name, label, index, len(selected))
                if generator is None or generator.checkpoint != checkpoint:
                    generator = None
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    with view.status('Loading checkpoint'):
                        generator = CheckpointGenerator(checkpoint, settings, device)
                with view.status('Generating continuation'):
                    row = generator.generate(prompt, seed, count)
                row['alias'] = name
                view.result(row)
                if output is not None:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with output.open('a', encoding='utf-8') as file:
                        file.write(json.dumps(row, ensure_ascii=False) + '\n')
                    view.notice(f'Saved to {output}')
            if args.prompt is not None:
                break
        if args.prompt is None:
            view.notice('Session finished.')
        return 0
    except (EOFError, KeyboardInterrupt, ModelSelectionCancelled):
        view.notice('Demo finished.')
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, yaml.YAMLError) as error:
        view.error(str(error))
        return 1
