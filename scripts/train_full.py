#!/usr/bin/env python3
"""Ejecuta secuencialmente la calibración o el entrenamiento de las tres variantes."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from src.training.full import relative_path, run_full


def experiment_configs(suite: dict, phase: str, all_seeds: bool, budget: str = 'course') -> list[dict]:
    """Deriva salidas únicas por fase, variante y seed desde configs/."""
    configs = []
    for seed in suite['seeds'] if all_seeds else suite['seeds'][:1]:
        for source in suite['configs']:
            cfg = copy.deepcopy(yaml.safe_load(relative_path(source).read_text()))
            cfg['experimento']['seed'] = seed
            tag = f"{cfg['experimento']['nombre']}_seed{seed}"
            limits = suite['budgets'][budget]
            cfg['entrenamiento']['max_train_seconds'] = limits['max_train_seconds']
            cfg['presupuesto']['total_seconds'] = limits['total_seconds']
            if phase == 'train' and 'epochs' in limits:
                cfg['entrenamiento']['epochs'] = limits['epochs']
            if budget != 'course' and phase == 'train':
                tag += f'_{budget}'
            if phase == 'calibrate':
                tag = f'calibrate_{tag}'
                overrides = suite['calibration']
                for key in ('max_steps', 'max_train_seconds', 'checkpoint_every'):
                    cfg['entrenamiento'][key] = overrides[key]
                for key in ('eval_every', 'max_batches'):
                    cfg['evaluacion'][key] = overrides[key]
            if phase == 'verify':
                tag = f'verify_{tag}'
                for section, values in suite['verification'].items():
                    cfg[section].update(values)
            cfg['experimento']['nombre'] = tag
            cfg['salida']['directory'] = f'runs/{tag}'
            configs.append(cfg)
    return configs


def main() -> None:
    """Lanza los procesos en serie; --resume continúa o salta los ya terminados."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/full_experiment.yaml'))
    parser.add_argument('--phase', choices=['verify', 'calibrate', 'train'], required=True)
    parser.add_argument('--budget', choices=['course', 'extended', 'dataset'], default='course')
    parser.add_argument('--all-seeds', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    suite = yaml.safe_load(relative_path(str(args.config)).read_text())
    for cfg in experiment_configs(suite, args.phase, args.all_seeds, args.budget):
        checkpoint = relative_path(cfg['salida']['directory']) / 'last.pt'
        print(f"\n=== {cfg['experimento']['nombre']} ===", flush=True)
        run_full(cfg, resume=checkpoint if args.resume and checkpoint.exists() else None)


if __name__ == '__main__':
    main()
