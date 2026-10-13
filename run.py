#!/usr/bin/env python3
"""Entrada única: --smoke (A), --config configs/overfit.yaml (B) o pilot_*.yaml (C)."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.training.probes import run_probe


def cargar_config(ruta: Path) -> dict:
    """Lee un YAML; los hiperparámetros se declaran exclusivamente en configs/."""
    with ruta.open(encoding='utf-8') as stream:
        return yaml.safe_load(stream)


def main() -> int:
    """Ejecuta una prueba y devuelve error cuando falla su criterio de aceptación."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/default.yaml'))
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    cfg = cargar_config(Path('configs/smoke.yaml') if args.smoke else args.config)
    report = run_probe(cfg)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
