#!/usr/bin/env python3
"""Calibra microbatches en procesos aislados y compara throughput y VRAM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from scripts.train_full import experiment_configs
from src.training.full import relative_path, run_full


def benchmark_configs(settings: dict) -> list[dict]:
    """Deriva configuraciones sin alterar los YAML de entrenamiento principal."""
    suite = yaml.safe_load(relative_path(settings['experiment_config']).read_text())
    configs = []
    for profile, overrides in settings['profiles'].items():
        if Path(profile).name != profile or profile in ('.', '..'):
            raise ValueError('Nombre de perfil inválido')
        if min(overrides['batch_size'], overrides['gradient_accumulation_steps']) < 1:
            raise ValueError('Batch y acumulación deben ser positivos')
        for cfg in experiment_configs(suite, 'calibrate', all_seeds=False):
            cfg['datos']['batch_size'] = overrides['batch_size']
            cfg['entrenamiento']['gradient_accumulation_steps'] = overrides['gradient_accumulation_steps']
            cfg['entrenamiento'].pop('epochs', None)
            tag = f"{cfg['experimento']['nombre']}_{profile}"
            cfg['experimento']['nombre'] = tag
            cfg['salida']['directory'] = str(relative_path(settings['output']) / tag)
            configs.append(cfg)
    if len({cfg['salida']['directory'] for cfg in configs}) != len(configs):
        raise ValueError('Los experimentos necesitan salidas diferentes')
    return configs


def summarize(configs: list[dict], output: Path) -> list[dict]:
    """Exporta resultados y marca informes parciales o fallidos sin declararlos ganadores."""
    rows = []
    for cfg in configs:
        root = relative_path(cfg['salida']['directory'])
        row = {'experiment': cfg['experimento']['nombre'], 'microbatch': cfg['datos']['batch_size'],
               'accumulation': cfg['entrenamiento']['gradient_accumulation_steps'],
               'status': 'missing', 'tokens_per_second': None, 'allocated_GiB': None,
               'reserved_GiB': None, 'skipped_updates': None, 'pipeline_seconds': None}
        if (root / 'report.json').exists():
            report = json.loads((root / 'report.json').read_text())
            enough = report['profile']['updates'] >= cfg['entrenamiento']['max_steps'] - cfg['entrenamiento']['profile_warmup_steps']
            row.update(status='ok' if enough and report['skipped_updates'] == 0 and report['config'] == cfg else 'check',
                       tokens_per_second=report['training_tokens_per_second'],
                       allocated_GiB=report['peak_allocated_bytes'] / 1024 ** 3,
                       reserved_GiB=report['peak_reserved_bytes'] / 1024 ** 3,
                       skipped_updates=report['skipped_updates'], pipeline_seconds=report['session_pipeline_seconds'])
        elif (root / 'failure.json').exists():
            failure = json.loads((root / 'failure.json').read_text())
            row.update(status='failed', error=failure)
        rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'comparison.json').write_text(json.dumps(rows, indent=2))
    print('\nExperiment                                         tok/s   allocated GiB  reserved GiB  skipped  status', flush=True)
    for row in rows:
        def number(key: str) -> str:
            """Formatea mediciones disponibles, conservando fallos como guiones."""
            value = row[key]
            return f'{value:.2f}' if isinstance(value, (float, int)) else '-'
        print(f"{row['experiment']:<49} {number('tokens_per_second'):>8} {number('allocated_GiB'):>14} {number('reserved_GiB'):>13} {number('skipped_updates'):>8}  {row['status']}", flush=True)
    return rows


def main() -> None:
    """Ejecuta cada calibración en un proceso nuevo, también después de un OOM."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/batch_calibration.yaml'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worker', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    config_path = relative_path(str(args.config))
    settings = yaml.safe_load(config_path.read_text())
    configs = benchmark_configs(settings)
    if args.dry_run:
        for cfg in configs:
            print(f"{cfg['experimento']['nombre']}: microbatch={cfg['datos']['batch_size']} accumulation={cfg['entrenamiento']['gradient_accumulation_steps']} output={cfg['salida']['directory']}")
        return
    if args.worker is not None:
        cfg = configs[args.worker]
        checkpoint = relative_path(cfg['salida']['directory']) / 'last.pt'
        run_full(cfg, resume=checkpoint if args.resume and checkpoint.exists() else None)
        return
    output = relative_path(settings['output'])
    if not args.report_only:
        # Rechazar una sobrescritura antes de iniciar cualquiera de los seis procesos.
        for cfg in configs:
            root = relative_path(cfg['salida']['directory'])
            if root.exists() and any(root.iterdir()) and not args.resume:
                parser.error(f'{root} ya existe; usar --resume o --report-only')
        for index, cfg in enumerate(configs):
            root = relative_path(cfg['salida']['directory'])
            if args.resume and (root / 'report.json').exists():
                report = json.loads((root / 'report.json').read_text())
                if report['config'] != cfg:
                    parser.error(f'{root}: configuración distinta; escoger otra salida en configs/')
                if report['step'] >= cfg['entrenamiento']['max_steps']:
                    continue
            command = [sys.executable, 'scripts/benchmark_batches.py', '--config', str(config_path), '--worker', str(index)]
            if args.resume:
                command.append('--resume')
            print(f"\n=== {cfg['experimento']['nombre']} ===", flush=True)
            result = subprocess.run(command, check=False)
            if result.returncode:
                print(f"Calibration failed ({result.returncode}); continuing with the next configuration.", flush=True)
    rows = summarize(configs, output)
    if not args.report_only and any(row['status'] != 'ok' for row in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
