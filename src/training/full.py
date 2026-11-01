"""Entrenamiento completo con acumulación, orden de datos y checkpoints reanudables."""
from __future__ import annotations

import copy
import json
import math
import platform
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from src.data.text import load_text
from src.models.factory import construir_modelo
from src.training.probes import evaluate, loss_for
from src.utils.seed import capturar_rng, fijar_seed, generador_datos, restaurar_rng
from src.utils.tracking import abrir_run, registrar


def relative_path(value: str) -> Path:
    """Rechaza rutas absolutas o que escapen del proyecto."""
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Usar rutas relativas dentro del proyecto')
    return path


class BatchStream:
    """Recorre una permutación por época sin reconstruir iteradores al reanudar."""

    def __init__(self, dataset: Any, batch_size: int, seed: int) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.generator = generador_datos(seed)
        self.order = torch.randperm(len(dataset), generator=self.generator)
        self.position = 0
        self.epoch = 0

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Devuelve un microbatch; la última tanda de una época puede ser menor."""
        if self.position == len(self.order):
            self.order = torch.randperm(len(self.dataset), generator=self.generator)
            self.position = 0
            self.epoch += 1
        ids = self.order[self.position:self.position + self.batch_size]
        self.position += len(ids)
        return tuple(tensor[ids] for tensor in self.dataset.tensors)

    def state_dict(self) -> dict:
        """Captura permutación, cursor y generador de la próxima época."""
        return {'order': self.order, 'position': self.position,
                'epoch': self.epoch, 'generator': self.generator.get_state()}

    def load_state_dict(self, state: dict) -> None:
        """Restaura un stream del mismo dataset verificado por hash."""
        self.order = state['order'].cpu()
        if len(self.order) != len(self.dataset):
            raise ValueError('Dataset distinto del checkpoint')
        self.position, self.epoch = state['position'], state['epoch']
        self.generator.set_state(state['generator'].cpu())


def learning_rate(cfg: dict, update: int, total: int) -> float:
    """Warmup real y coseno según actualizaciones aplicadas al optimizador activo."""
    warmup = cfg['lr_warmup_updates']
    if update < warmup:
        factor = (update + 1) / warmup
    else:
        progress = min(1.0, (update - warmup) / max(1, total - warmup - 1))
        floor = cfg['min_lr_ratio']
        factor = floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2
    return cfg['learning_rate'] * factor


def save_checkpoint(path: Path, state: dict) -> None:
    """Sustituye un checkpoint completo mediante rename atómico."""
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def resume_signature(cfg: dict) -> dict:
    """Permite ampliar tiempo disponible, conservando horizonte LR y experimento."""
    result = copy.deepcopy(cfg)
    result.pop('wandb', None)
    result.pop('presupuesto', None)
    result['entrenamiento'].pop('max_train_seconds', None)
    return result


def run_full(cfg: dict, resume: Path | None = None, stop_after: int | None = None) -> dict:
    """Entrena y guarda estado exacto; stop_after permite verificar interrupciones en tests."""
    session_start = time.perf_counter()
    output = relative_path(cfg['salida']['directory'])
    if resume is not None and relative_path(str(resume)).resolve() != (output / 'last.pt').resolve():
        raise ValueError('Reanudar desde last.pt en su directorio original; best.pt es para evaluación')
    if resume is None and output.exists() and any(output.iterdir()):
        raise FileExistsError(f'{output} contiene artefactos; usar --resume o una salida nueva')
    tcfg = cfg['entrenamiento']
    if tcfg['initial_loss_scale'] <= 0:
        raise ValueError('initial_loss_scale debe ser positivo')
    if min(tcfg['max_steps'], tcfg['gradient_accumulation_steps'], tcfg['checkpoint_every'],
           cfg['datos']['batch_size'], tcfg['log_every']) < 1:
        raise ValueError('Pasos, batch, acumulación e intervalos deben ser positivos')
    if not 0 <= tcfg['min_lr_ratio'] <= 1 or tcfg['lr_warmup_updates'] < 0:
        raise ValueError('Scheduler inválido')
    epochs = tcfg.get('epochs')
    if epochs is not None and (not isinstance(epochs, int) or epochs < 1):
        raise ValueError('epochs debe ser un entero positivo')
    if tcfg['max_train_seconds'] is None and epochs is None:
        raise ValueError('Sin límite de tiempo se requiere un objetivo epochs')
    if (tcfg['max_train_seconds'] is not None and tcfg['max_train_seconds'] <= 0) or cfg['evaluacion']['max_batches'] < 1 or cfg['evaluacion']['eval_every'] < 0:
        raise ValueError('Presupuesto o evaluación inválidos')
    if cfg['datos']['num_workers'] != 0:
        raise ValueError('El stream reanudable requiere num_workers=0')
    device = torch.device(tcfg['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA no disponible')
    if tcfg['precision'] not in ('fp32', 'fp16') or (tcfg['precision'] == 'fp16' and device.type != 'cuda'):
        raise ValueError('FP16 requiere CUDA; CPU usa fp32')
    saved = torch.load(relative_path(str(resume)), map_location='cpu', weights_only=True) if resume else None
    if saved and resume_signature(saved['config']) != resume_signature(cfg):
        raise ValueError('Configuración incompatible con reanudación (incluido max_steps)')
    if saved and saved.get('format') != 'full-v1':
        raise ValueError('Checkpoint sin soporte de reanudación exacta')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.json').write_text(json.dumps(cfg, indent=2))
    fijar_seed(cfg['experimento']['seed'])
    torch.set_num_threads(tcfg['cpu_threads'])
    train, validation, _, metadata = load_text(cfg)
    steps_per_epoch = math.ceil(math.ceil(len(train.dataset) / cfg['datos']['batch_size']) / tcfg['gradient_accumulation_steps'])
    target_steps = steps_per_epoch * epochs if epochs is not None else tcfg['max_steps']
    print(f'Training windows={len(train.dataset)}; target updates={target_steps}; epochs={epochs}', flush=True)
    model = construir_modelo(cfg).to(device)
    groups = [list(block.parameters()) for block in model.blocks] if model.diffusion else [[p for p in model.parameters() if p.requires_grad]]
    optimizers = [torch.optim.AdamW(params, lr=tcfg['learning_rate'], betas=tuple(tcfg['betas']), weight_decay=tcfg['weight_decay']) for params in groups]
    scaler = torch.amp.GradScaler('cuda', init_scale=tcfg['initial_loss_scale'], enabled=tcfg['precision'] == 'fp16')
    stream = BatchStream(train.dataset, cfg['datos']['batch_size'], cfg['experimento']['seed'])
    counts, token_counts = [0] * len(groups), [0] * len(groups)
    step = skipped = 0
    elapsed = 0.0
    curve: list[dict] = []
    initial: dict = {}
    best_score = float('inf')
    profile = {'seconds': 0.0, 'tokens': 0, 'updates': 0}
    run = abrir_run(cfg)
    try:
        if saved:
            if saved['metadata'] != metadata:
                raise ValueError('Cache de datos distinto del checkpoint')
            model.load_state_dict(saved['model'])
            for optimizer, state in zip(optimizers, saved['optimizers'], strict=True):
                optimizer.load_state_dict(state)
            scaler.load_state_dict(saved['scaler'])
            stream.load_state_dict(saved['stream'])
            counts, token_counts = saved['updates'], saved['tokens_per_block']
            step, skipped, elapsed = saved['step'], saved['skipped'], saved['elapsed_seconds']
            curve, initial, best_score, profile = saved['curve'], saved['initial'], saved['best_score'], saved['profile']
            expected_gpu = saved['environment']['gpu']
            actual_gpu = torch.cuda.get_device_name(device) if device.type == 'cuda' else None
            if saved['environment']['torch'] != str(torch.__version__) or expected_gpu != actual_gpu:
                raise ValueError('Reanudar exactamente requiere la misma versión de PyTorch y GPU')
            restaurar_rng(saved['rng'])
        else:
            initial = evaluate(model, validation, cfg, device)
            registrar(run, {f'initial/{k}': v for k, v in initial.items()}, 0)
        environment = {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                       'python': platform.python_version(), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None}
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        setup_seconds = started - session_start
        # El checkpoint es la fuente de verdad si hubo un fallo después de escribir el log.
        log_path = output / 'steps.jsonl'
        if saved and log_path.exists():
            lines = log_path.read_text().splitlines()
            committed = []
            for index, line in enumerate(lines):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    if index != len(lines) - 1:
                        raise
                    break
                if row['step'] <= step:
                    committed.append(line)
            log_path.write_text(''.join(line + '\n' for line in committed))
        (output / 'failure.json').unlink(missing_ok=True)

        def checkpoint() -> dict:
            return {'format': 'full-v1', 'config': cfg, 'model': model.state_dict(),
                    'optimizers': [opt.state_dict() for opt in optimizers], 'scaler': scaler.state_dict(),
                    'stream': stream.state_dict(), 'rng': capturar_rng(), 'metadata': metadata,
                    'step': step, 'updates': counts, 'tokens_per_block': token_counts,
                    'skipped': skipped, 'elapsed_seconds': elapsed + time.perf_counter() - started,
                    'curve': curve, 'initial': initial, 'best_score': best_score,
                    'profile': profile, 'environment': environment}

        def validate() -> dict:
            nonlocal best_score
            metrics = evaluate(model, validation, cfg, device)
            curve.append({'step': step, **metrics})
            registrar(run, {f'validation/{k}': v for k, v in metrics.items()}, step)
            score = metrics['loss_sigma_max'] if model.diffusion else metrics['loss']
            if score < best_score:
                best_score = score
                save_checkpoint(output / 'best.pt', checkpoint())
            return metrics

        if not saved:
            save_checkpoint(output / 'last.pt', checkpoint())
        with (output / 'steps.jsonl').open('a', encoding='utf-8') as log:
            while step < target_steps:
                time_limit_reached = tcfg['max_train_seconds'] is not None and elapsed + time.perf_counter() - started >= tcfg['max_train_seconds']
                if time_limit_reached or (stop_after is not None and step >= stop_after):
                    break
                block = step % len(groups)
                optimizer = optimizers[block]
                for opt in optimizers:
                    opt.zero_grad(set_to_none=True)
                lr = learning_rate(tcfg, counts[block], math.ceil(target_steps / len(groups)))
                for group in optimizer.param_groups:
                    group['lr'] = lr
                batches = []
                for _ in range(tcfg['gradient_accumulation_steps']):
                    batches.append(stream.next())
                    if epochs is not None and stream.position == len(stream.order):
                        break  # Conservar el último batch parcial sin repetir datos de la próxima época.
                # Peso por token, incluyendo microbatches parciales y padding.
                n = sum(int((valid[:, :-1] if model.diffusion else valid[:, 1:]).sum()) for _, valid in batches)
                if not n:
                    raise ValueError('Actualización sin tokens válidos')
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                tick = time.perf_counter()
                model.train()
                aggregate = 0.0
                for tokens, valid in batches:
                    context = torch.autocast('cuda', dtype=torch.float16) if scaler.is_enabled() else nullcontext()
                    with context:
                        loss, count = loss_for(model, tokens.to(device), valid.to(device), block, cfg)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Pérdida no finita')
                    scaler.scale(loss * (count / n)).backward()
                    aggregate += float(loss.detach()) * count / n
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(groups[block], tcfg['gradient_clip'])
                if not torch.isfinite(norm) and not scaler.is_enabled():
                    raise FloatingPointError('Gradiente no finito')
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                updated = scaler.get_scale() >= old_scale
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                seconds = time.perf_counter() - tick
                if updated:
                    counts[block] += 1
                    token_counts[block] += n
                else:
                    skipped += 1
                step += 1
                if step > tcfg['profile_warmup_steps']:
                    profile['seconds'] += seconds
                    profile['tokens'] += n
                    profile['updates'] += 1
                row = {'step': step, 'block': block, 'loss': aggregate, 'lr': lr,
                       'gradient_norm': float(norm) if torch.isfinite(norm) else None,
                       'updated': updated, 'tokens': n, 'seconds': seconds, 'epoch': stream.epoch}
                log.write(json.dumps(row) + '\n')
                log.flush()
                registrar(run, {f'train/block_{block}/{k}': v for k, v in row.items()}, step)
                if step % tcfg['log_every'] == 0:
                    print(f'step={step} block={block} loss={aggregate:.4f} lr={lr:.3g}', flush=True)
                if cfg['evaluacion']['eval_every'] and step % cfg['evaluacion']['eval_every'] == 0:
                    validate()
                if step % tcfg['checkpoint_every'] == 0:
                    save_checkpoint(output / 'last.pt', checkpoint())
        # Guardar antes y después: evaluación restaura RNG y un fallo no pierde el entrenamiento.
        save_checkpoint(output / 'last.pt', checkpoint())
        final = validate() if not curve or curve[-1]['step'] != step else {k: v for k, v in curve[-1].items() if k != 'step'}
        save_checkpoint(output / 'last.pt', checkpoint())
        report = {'config': cfg, 'environment': environment, 'data': metadata,
                  'step': step, 'completed_steps': step >= target_steps,
                  'target_steps': target_steps, 'training_windows': len(train.dataset),
                  'epochs_completed': stream.epoch + stream.position / len(stream.order),
                  'updates_per_block': counts, 'tokens_per_block': token_counts, 'skipped_updates': skipped,
                  'initial': initial, 'validation': final, 'validation_curve': curve,
                  'elapsed_seconds': elapsed + time.perf_counter() - started,
                  'session_setup_seconds': setup_seconds,
                  'session_pipeline_seconds': time.perf_counter() - session_start,
                  'training_tokens_per_second': profile['tokens'] / profile['seconds'] if profile['seconds'] else None,
                  'profile': profile,
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
                  'peak_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0,
                  'parameters_total': sum(p.numel() for p in model.parameters()),
                  'parameters_trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
                  'inactive_blocks': [b for b in range(len(groups)) if initial[f'loss_block_{b}'] == 0],
                  'budget_met': (setup_seconds + elapsed + time.perf_counter() - started < cfg['presupuesto']['total_seconds']) if cfg['presupuesto']['total_seconds'] is not None else None}
        (output / 'report.json').write_text(json.dumps(report, indent=2))
        registrar(run, {'result/step': step, 'result/elapsed_seconds': report['elapsed_seconds'],
                        'profile/peak_allocated_bytes': report['peak_allocated_bytes'],
                        'profile/peak_reserved_bytes': report['peak_reserved_bytes'],
                        'profile/tokens_per_second': report['training_tokens_per_second'],
                        'profile/updates_per_block': counts, 'profile/tokens_per_block': token_counts,
                        'profile/skipped_updates': skipped, 'checks/budget_met': report['budget_met']}, step)
        print(f"Saved {output}/last.pt; steps={step}; updates={counts}", flush=True)
        return report
    except Exception as error:
        (output / 'failure.json').write_text(json.dumps({'type': type(error).__name__, 'message': str(error)}))
        raise
    finally:
        if run is not None:
            run.finish()
