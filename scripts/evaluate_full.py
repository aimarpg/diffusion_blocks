#!/usr/bin/env python3
"""Evalúa checkpoints sobre historias reservadas y exporta continuaciones para revisión ciega."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from contextlib import nullcontext

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from src.data.text import load_text
from src.models.factory import construir_modelo
from src.training.full import relative_path
from src.training.probes import evaluate
from src.utils.seed import seed_temporal


def repetition(ids: list[int], n: int) -> float:
    """Fracción de n-gramas repetidos dentro de una continuación."""
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def run_evaluation(settings: dict, resume: bool = False) -> None:
    """Mantiene los mismos prompts, seeds y sampler en todos los checkpoints."""
    output = relative_path(settings['output'])
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise FileExistsError('La salida de evaluación ya contiene artefactos; usar --resume')
        if json.loads((output / 'config.json').read_text()) != settings:
            raise ValueError('La configuración de evaluación difiere de la guardada')
    device = torch.device(settings['device'])
    if settings['precision'] not in ('fp32', 'fp16') or (settings['precision'] == 'fp16' and device.type != 'cuda'):
        raise ValueError('FP16 requiere CUDA; CPU usa fp32')
    torch.set_num_threads(settings['cpu_threads'])
    paths = [relative_path(value) for value in settings['checkpoints']]
    # Verificar compatibilidad ANTES de crear resultados parciales.
    signatures = []
    for path in paths:
        saved = torch.load(path, map_location='cpu', weights_only=True)
        signatures.append((saved['metadata'], saved['config']['datos'],
                           {k: v for k, v in saved['config']['modelo'].items() if k not in ('method', 'noise')}))
        del saved
    if not signatures or any(item != signatures[0] for item in signatures[1:]):
        raise ValueError('Los checkpoints deben compartir cache, datos y arquitectura base')
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('Indicar checkpoints distintos')
    if settings['prompt_tokens'] < 1 or settings['test_stories'] < 1 or settings['batch_size'] < 1 or settings['max_batches'] < 0:
        raise ValueError('Configuración de datos inválida')
    if settings['prompt_count'] < 1 or settings['max_new_tokens'] < 1 or not settings['seeds']:
        raise ValueError('Prompts, tokens y seeds deben ser positivos/no vacíos')
    checkpoint_hashes = {}
    for path in paths:
        with path.open('rb') as file:
            checkpoint_hashes[str(path)] = hashlib.file_digest(file, 'sha256').hexdigest()
    hash_path = output / 'checkpoint_hashes.json'
    if resume and hash_path.exists() and json.loads(hash_path.read_text()) != checkpoint_hashes:
        raise ValueError('Los checkpoints han cambiado desde la evaluación anterior')
    output.mkdir(parents=True, exist_ok=True)
    hash_path.write_text(json.dumps(checkpoint_hashes, indent=2))
    (output / 'config.json').write_text(json.dumps(settings, indent=2))
    summaries, blind, key = [], [], []
    existing = {}
    generation_log = output / 'generations.jsonl'
    if resume and generation_log.exists():
        lines = generation_log.read_text().splitlines()
        valid_lines = []
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise
                break
            existing[(row['checkpoint'], row['prompt_id'], row['seed'])] = row
            valid_lines.append(line)
        generation_log.write_text(''.join(line + '\n' for line in valid_lines))
    prompts = None
    for model_index, path in enumerate(paths):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        cfg = copy.deepcopy(saved['config'])
        cfg['datos'].update(path=settings['data_path'], test_stories=settings['test_stories'], batch_size=settings['batch_size'])
        cfg['evaluacion'].update(seed=settings['evaluation_seed'], max_batches=settings['max_batches'])
        cfg['modelo']['noise'].update(steps_per_block=settings['steps_per_block'], clamp=settings['clamp'])
        _, loader, tokenizer, metadata = load_text(cfg, validation_split='test')
        if metadata != saved['metadata']:
            raise ValueError('El cache de evaluación no coincide con el de entrenamiento')
        cfg['evaluacion']['max_batches'] = settings['max_batches'] or len(loader)
        if prompts is None:
            stories = json.loads((relative_path(settings['data_path']) / 'test.json').read_text())[:settings['test_stories']]
            # Prompts de historias distintas, sin seleccionar según calidad del modelo.
            prompts = [story[:settings['prompt_tokens']] for story in stories
                       if len(story) > settings['prompt_tokens']][:settings['prompt_count']]
            if len(prompts) != settings['prompt_count']:
                raise ValueError('No hay suficientes historias para los prompts reservados')
            (output / 'prompts.json').write_text(json.dumps(prompts))
        model = construir_modelo(cfg).to(device).eval()
        model.load_state_dict(saved['model'])
        model_id = saved['config']['experimento']['nombre']
        del saved
        metrics = evaluate(model, loader, cfg, device)
        model_records = []
        # Baseline argmax es determinista: evitar copias idénticas por cada seed.
        for seed in settings['seeds'] if model.diffusion else settings['seeds'][:1]:
            for prompt_index, ids in enumerate(prompts):
                previous = existing.get((str(path), prompt_index, seed))
                prompt = torch.tensor([ids], device=device)
                if previous is None:
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    context = torch.autocast('cuda', dtype=torch.float16) if settings['precision'] == 'fp16' else nullcontext()
                    with seed_temporal(seed), context:
                        generated = model.generate(prompt, settings['max_new_tokens'])[0].tolist()[len(ids):]
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    seconds = time.perf_counter() - start
                    eos = cfg['datos']['eos_id']
                    eos_index = generated.index(eos) if eos in generated else None
                    visible = generated[:eos_index] if eos_index is not None else generated
                    identity = f'{model_id}:{prompt_index}:{seed}'
                    blind_id = hashlib.sha256(identity.encode()).hexdigest()[:16]
                    row = {'model': model_id, 'checkpoint': str(path), 'prompt_id': prompt_index,
                           'seed': seed, 'prompt': tokenizer.decode(ids), 'continuation': tokenizer.decode(visible),
                           'token_ids': generated, 'seconds': seconds, 'generated_tokens': len(generated),
                           'first_eos_index': eos_index, 'empty': not tokenizer.decode(visible).strip(),
                           'repetition_4gram': repetition(visible, 4), 'blind_id': blind_id}
                else:
                    row = previous
                    blind_id = row['blind_id']
                model_records.append(row)
                blind.append({'id': blind_id, 'prompt': row['prompt'], 'continuation': row['continuation'],
                              'coherence': '', 'grammar': '', 'relevance': '', 'consistency': '', 'notes': ''})
                key.append({'id': blind_id, 'model': model_id, 'prompt_id': prompt_index, 'seed': seed})
                # Persistir cada generación: esta fase puede durar más que el entrenamiento.
                if previous is None:
                    with generation_log.open('a') as file:
                        file.write(json.dumps(row) + '\n')
                print(f'{model_id}: prompt {prompt_index + 1}/{len(prompts)} seed={seed}', flush=True)
        summaries.append({'model': model_id, 'checkpoint': str(path), 'metrics': metrics,
                          'generation_count': len(model_records),
                          'tokens_per_second': sum(r['generated_tokens'] for r in model_records) / sum(r['seconds'] for r in model_records),
                          'mean_repetition_4gram': sum(r['repetition_4gram'] for r in model_records) / len(model_records),
                          'empty_fraction': sum(r['empty'] for r in model_records) / len(model_records)})
        (output / 'summary.json').write_text(json.dumps(summaries, indent=2))
        del model
    blind.sort(key=lambda row: row['id'])
    with (output / 'blind_ratings.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(blind[0]))
        writer.writeheader()
        writer.writerows(blind)
    (output / 'blind_key.json').write_text(json.dumps(key, indent=2))


def main() -> None:
    """Lee todos los hiperparámetros de evaluación desde configs/."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/evaluate_full.yaml'))
    parser.add_argument('--budget', choices=['course', 'extended', 'dataset'])
    parser.add_argument('--all-seeds', action='store_true')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    settings = yaml.safe_load(relative_path(str(args.config)).read_text())
    if args.budget or args.all_seeds:
        from scripts.train_full import experiment_configs
        suite = yaml.safe_load(relative_path(settings['experiment_config']).read_text())
        configs = experiment_configs(suite, 'train', args.all_seeds, args.budget or 'course')
        settings['checkpoints'] = [str(relative_path(c['salida']['directory']) / 'best.pt') for c in configs]
        if args.budget and args.budget != 'course':
            settings['output'] += f'_{args.budget}'
        if args.all_seeds:
            settings['output'] += '_all_seeds'
    if args.quick:
        settings.update(settings['quick'])
        settings['output'] += '_quick'
    run_evaluation(settings, resume=args.resume)


if __name__ == '__main__':
    main()
