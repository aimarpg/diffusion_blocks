#!/usr/bin/env python3
"""Compara métricas medidas de C; rechaza ejecuciones con condiciones incompatibles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare(baseline: dict, diffusion: dict) -> dict:
    """Devuelve ratios de tiempo y memoria solo para una pareja comparable."""
    if baseline['method'] != 'baseline' or diffusion['method'] != 'diffusion':
        raise ValueError('Se esperan informes baseline y diffusion, en ese orden')
    if any(report['prueba'] != 'C' or not report['passed'] for report in (baseline, diffusion)):
        raise ValueError('Ambas pruebas C deben haber pasado')
    for section in ('datos', 'entrenamiento', 'evaluacion', 'presupuesto'):
        if baseline['config'][section] != diffusion['config'][section]:
            raise ValueError(f'Configuraciones distintas: {section}')
    for key in ('seed',):
        if baseline['config']['experimento'][key] != diffusion['config']['experimento'][key]:
            raise ValueError(f'Distinto {key}')
    left = {k: v for k, v in baseline['config']['modelo'].items() if k != 'method'}
    right = {k: v for k, v in diffusion['config']['modelo'].items() if k != 'method'}
    if left != right or baseline['data'] != diffusion['data'] or baseline['environment'] != diffusion['environment']:
        raise ValueError('Arquitectura, datos o entorno distintos')
    if baseline['measured_steps'] != diffusion['measured_steps']:
        raise ValueError('Ventanas de medición distintas')
    return {'baseline_step_median_seconds': baseline['step_median_seconds'], 'diffusion_step_median_seconds': diffusion['step_median_seconds'], 'step_speed_ratio_baseline_over_diffusion': baseline['step_median_seconds'] / diffusion['step_median_seconds'], 'allocated_memory_ratio_baseline_over_diffusion': baseline['peak_allocated_bytes'] / diffusion['peak_allocated_bytes'], 'reserved_memory_ratio_baseline_over_diffusion': baseline['peak_reserved_bytes'] / diffusion['peak_reserved_bytes'], 'baseline_updates': baseline['updates_per_block'], 'diffusion_updates': diffusion['updates_per_block'], 'baseline_trainable': baseline['parameters_trainable'], 'diffusion_trainable': diffusion['parameters_trainable'], 'interpretation': 'Un paso diffusion actualiza un bloque; estos ratios no comparan calidad ni tiempo hasta convergencia.'}


def main() -> None:
    """Lee informes JSON y muestra una comparación sin ejecutar entrenamiento."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--diffusion', type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.baseline.read_text()), json.loads(args.diffusion.read_text()))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
