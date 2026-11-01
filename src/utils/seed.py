"""Semillas aleatorias, fijadas en UN SOLO SITIO.

Si te encuentras llamando a manual_seed en tres ficheros distintos, es que
esta funcion no se esta usando. El checker avisa de eso.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import os
import random

import numpy as np
import torch


def fijar_seed(seed: int, determinista: bool = True) -> None:
    """Fija las semillas de las tres fuentes de aleatoriedad del proyecto.

    Args:
        seed: la semilla. Va en configs/, no aqui.
        determinista: fuerza los kernels deterministas de cuDNN. Cuesta algo de
            velocidad; a la escala de la regla 5/20 no se nota y a cambio dos
            ejecuciones dan el mismo numero.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if determinista:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@contextmanager
def seed_temporal(seed: int) -> Iterator[None]:
    """Fija ruido de evaluación y restaura RNG de Python, NumPy, CPU y CUDA."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            fijar_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def capturar_rng() -> dict:
    """Captura RNG serializable con torch.load(weights_only=True)."""
    state = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': (state[0], state[1].tolist(), state[2], state[3], state[4]),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restaurar_rng(state: dict) -> None:
    """Restaura el RNG global después de construir modelo y optimizadores."""
    random.setstate(state['python'])
    ns = state['numpy']
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), *ns[2:]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([item.cpu() for item in state['cuda']])


def generador_datos(seed: int) -> torch.Generator:
    """Crea un RNG independiente para el orden de datos reproducible."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
