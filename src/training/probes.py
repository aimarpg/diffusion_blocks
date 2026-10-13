"""Pruebas A/B/C: entrenamiento acotado, evaluación fija y perfilado sincronizado."""
from __future__ import annotations

import json
import math
import platform
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data.text import load_text
from src.models.language import LanguageModel
from src.models.factory import construir_modelo
from src.utils.seed import fijar_seed, seed_temporal
from src.utils.tracking import abrir_run, registrar


def loss_for(model: LanguageModel, tokens: torch.Tensor, valid: torch.Tensor, block: int, cfg: dict) -> tuple[torch.Tensor, int]:
    """Calcula CE sobre tokens válidos; baseline desplaza labels, denoising no."""
    if model.diffusion:
        inputs, labels, mask = tokens[:, :-1], tokens[:, :-1], valid[:, :-1]
        logits = model(inputs, block=block)
    else:
        inputs, labels, mask = tokens[:, :-1], tokens[:, 1:], valid[:, 1:]
        logits = model(inputs)
    count = int(mask.sum())
    if not count:
        raise ValueError('Batch sin tokens objetivo válidos')
    labels = labels.masked_fill(~mask, -100)
    loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1))
    return loss, count


@torch.no_grad()
def evaluate(model: LanguageModel, loader: DataLoader, cfg: dict, device: torch.device) -> dict[str, float]:
    """Evalúa bloques con ruido fijo, restaurando el estado RNG del entrenamiento."""
    was_training = model.training
    model.eval()
    result = {}
    with seed_temporal(cfg['evaluacion']['seed']):
        for block in range(len(model.blocks)):
            total, count = 0.0, 0
            for index, (tokens, valid) in enumerate(loader):
                if index >= cfg['evaluacion']['max_batches']:
                    break
                loss, n = loss_for(model, tokens.to(device), valid.to(device), block, cfg)
                total += float(loss) * n
                count += n
            if not count:
                raise ValueError('Evaluación sin tokens')
            result[f'loss_block_{block}'] = total / count
    result['loss'] = sum(result.values()) / len(result)
    if not model.diffusion:
        result['perplexity'] = math.exp(result['loss'])
    model.train(was_training)
    return result


def _sync(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def run_probe(cfg: dict) -> dict:
    """Ejecuta una prueba completa y escribe informe JSON, muestra y checkpoint."""
    started = time.perf_counter()
    output = Path(cfg['salida']['directory'])
    if output.is_absolute():
        raise ValueError('La salida debe ser una ruta relativa al proyecto')
    output.mkdir(parents=True, exist_ok=True)
    for name in ('report.json', 'failure.json'):
        (output / name).unlink(missing_ok=True)
    run = None
    try:
        fijar_seed(cfg['experimento']['seed'])
        tcfg = cfg['entrenamiento']
        device = torch.device(tcfg['device'])
        if cfg['experimento']['prueba'] == 'C' and device.type != 'cuda':
            raise ValueError('C debe medirse en CUDA; utilizar A/B para CPU')
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('La prueba C requiere CUDA. Ejecutarla en Colab/T4; no se sustituye por CPU.')
        if tcfg['precision'] not in ('fp32', 'fp16') or (tcfg['precision'] == 'fp16' and device.type != 'cuda'):
            raise ValueError('FP16 requiere CUDA; CPU usa fp32')
        if not 0 <= tcfg['warmup_steps'] < tcfg['max_steps'] or tcfg['log_every'] < 1:
            raise ValueError('Calentamiento o número de pasos inválido')
        if not 0 <= cfg['presupuesto']['reserve_fraction'] < 1:
            raise ValueError('La reserva debe estar entre 0 y 1')
        torch.set_num_threads(tcfg['cpu_threads'])
        run = abrir_run(cfg)
        train, validation, tokenizer, metadata = load_text(cfg)
        model = construir_modelo(cfg).to(device)
        groups = [list(block.parameters()) for block in model.blocks] if model.diffusion else [[p for p in model.parameters() if p.requires_grad]]
        optimizers = [torch.optim.AdamW(params, lr=tcfg['learning_rate'], weight_decay=tcfg['weight_decay'], betas=tuple(tcfg['betas'])) for params in groups]
        scaler = (torch.amp.GradScaler('cuda', enabled=tcfg['precision'] == 'fp16')
                  if hasattr(torch.amp, 'GradScaler')
                  else torch.cuda.amp.GradScaler(enabled=tcfg['precision'] == 'fp16'))
        initial = evaluate(model, train if cfg['experimento']['prueba'] == 'B' else validation, cfg, device)
        before_validation = evaluate(model, validation, cfg, device)
        counts = [0] * len(groups)
        token_counts = [0] * len(groups)
        skipped = 0
        timings, measured_tokens, history = [], 0, []
        iterator = iter(train)
        peak_allocated = peak_reserved = 0
        training_start = time.perf_counter()
        for step in range(tcfg['max_steps']):
            if time.perf_counter() - training_start >= tcfg['max_train_seconds']:
                break
            if step == tcfg['warmup_steps'] and device.type == 'cuda':
                _sync(device)
                torch.cuda.reset_peak_memory_stats(device)
            # Ronda equilibrada: los estados de todos los optimizadores se crean en warmup.
            block = step % len(groups)
            optimizer = optimizers[block]
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            model.train()
            _sync(device)
            tick = time.perf_counter()
            try:
                tokens, valid = next(iterator)
            except StopIteration:
                iterator = iter(train)
                tokens, valid = next(iterator)
            tokens, valid = tokens.to(device), valid.to(device)
            context = torch.autocast('cuda', dtype=torch.float16) if tcfg['precision'] == 'fp16' else nullcontext()
            with context:
                loss, n = loss_for(model, tokens, valid, block, cfg)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Pérdida no finita en bloque {block}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(groups[block], tcfg['gradient_clip'])
            if not torch.isfinite(gradient_norm):
                if not scaler.is_enabled():
                    raise FloatingPointError('Gradientes no finitos')
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            updated = scaler.get_scale() >= old_scale
            _sync(device)
            elapsed = time.perf_counter() - tick
            if updated:
                counts[block] += 1
                token_counts[block] += n
            else:
                skipped += 1
            if step >= tcfg['warmup_steps']:
                timings.append(elapsed)
                measured_tokens += n
            row = {'step': step + 1, 'block': block, 'loss': float(loss.detach()), 'gradient_norm': float(gradient_norm) if torch.isfinite(gradient_norm) else None, 'seconds': elapsed, 'updated': updated}
            history.append(row)
            registrar(run, {f'train/{key}': value for key, value in row.items()}, step + 1)
            if (step + 1) % tcfg['log_every'] == 0:
                print(f"paso {step + 1}/{tcfg['max_steps']} bloque={block} loss={row['loss']:.4f} s={elapsed:.3f}", flush=True)
        _sync(device)
        training_seconds = time.perf_counter() - training_start
        if device.type == 'cuda':
            peak_allocated = torch.cuda.max_memory_allocated(device)
            peak_reserved = torch.cuda.max_memory_reserved(device)
        after_train = evaluate(model, train if cfg['experimento']['prueba'] == 'B' else validation, cfg, device)
        after_validation = evaluate(model, validation, cfg, device)
        checkpoint = output / 'checkpoint.pt'
        torch.save({'config': cfg, 'model': model.state_dict(), 'optimizers': [opt.state_dict() for opt in optimizers], 'scaler': scaler.state_dict(), 'updates': counts, 'metadata': metadata}, checkpoint)
        prompt_ids = tokenizer.encode(cfg['generacion']['prompt'])
        if not prompt_ids:
            raise ValueError('Prompt vacío')
        prompt = torch.tensor([prompt_ids], device=device)
        generation_start = time.perf_counter()
        with seed_temporal(cfg['generacion']['seed']):
            generated = model.generate(prompt, cfg['generacion']['max_new_tokens'])
        _sync(device)
        generation_seconds = time.perf_counter() - generation_start
        # La recarga se valida en CPU para no duplicar el modelo completo en la T4.
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        restored = construir_modelo(saved['config'])
        restored.load_state_dict(saved['model'])
        reload_equal = all(torch.equal(value.cpu(), restored.state_dict()[key]) for key, value in model.state_dict().items())
        if not reload_equal:
            raise AssertionError('Checkpoint distinto tras recarga')
        # A/B validan además el sampler determinista con el mismo estado RNG.
        generation_equal = None
        if cfg['experimento']['prueba'] in ('A', 'B'):
            with seed_temporal(cfg['generacion']['seed']):
                actual = restored.generate(prompt.cpu(), cfg['generacion']['max_new_tokens'])
            generation_equal = bool(torch.equal(generated.cpu(), actual))
            if not generation_equal:
                raise AssertionError('Generación distinta tras recarga')
        sample = tokenizer.decode(generated[0].tolist())
        (output / 'sample.txt').write_text(sample, encoding='utf-8')
        (output / 'steps.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in history))
        ordered = sorted(timings)
        median = statistics.median(ordered) if ordered else None
        p95 = ordered[min(len(ordered) - 1, math.ceil(len(ordered) * .95) - 1)] if ordered else None
        registrar(run, {**{f'initial/{key}': value for key, value in initial.items()},
                        **{f'final/{key}': value for key, value in after_train.items()},
                        **{f'validation/{key}': value for key, value in after_validation.items()},
                        'profile/step_median_seconds': median, 'profile/step_p95_seconds': p95,
                        'profile/tokens_per_second': measured_tokens / sum(timings) if timings else None,
                        'profile/peak_allocated_bytes': peak_allocated,
                        'profile/peak_reserved_bytes': peak_reserved,
                        'profile/updates_per_block': counts, 'profile/skipped_updates': skipped,
                        'model/parameters_total': sum(p.numel() for p in model.parameters()),
                        'model/parameters_trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
                        'checks/checkpoint_equal': reload_equal,
                        'checks/generation_equal': generation_equal})
        if run is not None:
            run.finish()
            run = None
        total_seconds = time.perf_counter() - started
        overhead = total_seconds - training_seconds
        available = max(0, cfg['presupuesto']['total_seconds'] * (1 - cfg['presupuesto']['reserve_fraction']) - overhead)
        estimated_steps = int(available / p95) if p95 else None
        decreases = [after_train[f'loss_block_{b}'] < initial[f'loss_block_{b}'] * (1 - cfg['evaluacion']['minimum_relative_drop']) for b in range(len(groups))]
        report = {
            'prueba': cfg['experimento']['prueba'],
            'method': cfg['modelo']['method'],
            'config': cfg,
            'data': metadata,
            'environment': {
                'torch': str(torch.__version__),
                'cuda': torch.version.cuda,
                'cudnn': torch.backends.cudnn.version(),
                'python': platform.python_version(),
                'device': str(device),
                'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            },
            'parameters_total': sum(p.numel() for p in model.parameters()),
            'parameters_trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'initial': initial,
            'final': after_train,
            'validation_before': before_validation,
            'validation_after': after_validation,
            'updates_per_block': counts,
            'tokens_per_block': token_counts,
            'skipped_updates': skipped,
            'measured_steps': len(timings),
            'step_median_seconds': median,
            'step_p95_seconds': p95,
            'useful_tokens_per_second': measured_tokens / sum(timings) if timings else None,
            'peak_allocated_bytes': peak_allocated,
            'peak_reserved_bytes': peak_reserved,
            'training_seconds': training_seconds,
            'total_seconds': total_seconds,
            'generation_seconds': generation_seconds,
            'estimated_steps_in_budget': estimated_steps,
            'checkpoint_equal': reload_equal,
            'generation_equal': generation_equal,
            'loss_decreased_per_block': decreases,
            'sample': sample,
        }
        report['passed'] = all(counts) and reload_equal and (total_seconds < cfg['presupuesto']['total_seconds'])
        if cfg['experimento']['prueba'] == 'B':
            report['passed'] = report['passed'] and all(decreases)
        if cfg['experimento']['prueba'] == 'C':
            report['passed'] = report['passed'] and len(timings) >= tcfg['minimum_measured_steps'] and skipped == 0
        (output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"loss: {after_train['loss']:.4f}\npassed: {report['passed']}\ninforme: {output / 'report.json'}", flush=True)
        return report
    except Exception as error:
        registrar(run, {'result/passed': False, 'result/error': str(error)})
        (output / 'failure.json').write_text(json.dumps({'error': type(error).__name__, 'message': str(error), 'config': cfg}, indent=2))
        raise
    finally:
        if run is not None:
            run.finish()
